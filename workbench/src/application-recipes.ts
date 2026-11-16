export type ApplicationCategory = "geometry" | "analysis" | "features" | "depth" | "3d" | "models";
export type ApplicationAvailability = "ready" | "specialized" | "setup";

export interface ApplicationStep {
  id: string;
  title: string;
  publicApi: string;
  componentPatterns: string[];
  parents: string[];
  note?: string;
}

export interface ApplicationRecipe {
  id: string;
  index: string;
  title: string;
  summary: string;
  detail: string;
  audience: string;
  inputCount: 1 | 2;
  inputLabel: string;
  outputs: string[];
  categories: ApplicationCategory[];
  availability: ApplicationAvailability;
  stability: "stable" | "experimental";
  route?: string;
  runtimeNote: string;
  visual: "projection" | "edges" | "regions" | "features" | "planes" | "pose" | "stereo" | "depth" | "segmentation" | "reconstruction";
  featured?: boolean;
  steps: ApplicationStep[];
  python: string;
}

const loadSingle = `import numpy as np
from PIL import Image

erp = np.asarray(Image.open("panorama.jpg").convert("RGB"), dtype=np.float32) / 255.0`;

const loadPair = `import numpy as np
from PIL import Image

erp_a = np.asarray(Image.open("panorama_a.jpg").convert("RGB"), dtype=np.float32) / 255.0
erp_b = np.asarray(Image.open("panorama_b.jpg").convert("RGB"), dtype=np.float32) / 255.0`;

export const APPLICATION_RECIPES: ApplicationRecipe[] = [
  {
    id: "gnomonic-view", index: "01", title: "Câmera virtual na esfera",
    summary: "Abra uma vista gnomônica sem alterar a ERP original.",
    detail: "Controle direção, campo de visão e resolução com suporte geométrico explícito.",
    audience: "Geometria · inspeção · pré-processamento", inputCount: 1, inputLabel: "1 ERP RGB",
    outputs: ["vista projetada", "suporte"], categories: ["geometry"], availability: "ready", stability: "stable",
    runtimeNote: "Pronto na instalação atual", visual: "projection", featured: true,
    steps: [{ id: "project", title: "Projeção gnomônica", publicApi: "panorai.geometry.equirectangular_to_gnomonic", componentPatterns: ["equirectangular_to_gnomonic"], parents: ["input-a"] }],
    python: `${loadSingle}

from panorai.geometry import GnomonicSpec, equirectangular_to_gnomonic

view = equirectangular_to_gnomonic(
    erp,
    GnomonicSpec(output_shape_hw=(1024, 1024)),
    interpolation="bilinear",
)`,
  },
  {
    id: "spherical-edges", index: "02", title: "Bordas e estrutura esférica",
    summary: "Suavize ruído e detecte contornos sem quebrar a costura.",
    detail: "Um fluxo curto para inspeção estrutural, máscaras e etapas posteriores de visão.",
    audience: "Pré-processamento · análise de cena", inputCount: 1, inputLabel: "1 ERP RGB",
    outputs: ["raster filtrado", "bordas"], categories: ["analysis"], availability: "specialized", stability: "stable",
    route: "/single-view.html?experiment=filters-edges", runtimeNote: "Pronto na bancada Single View", visual: "edges",
    steps: [
      { id: "blur", title: "Gaussian blur esférico", publicApi: "panorai.ops.spherical.GaussianBlur", componentPatterns: ["GaussianBlur"], parents: ["input-a"] },
      { id: "canny", title: "Canny esférico", publicApi: "panorai.ops.spherical.Canny", componentPatterns: ["Canny", "canny"], parents: ["blur"] },
    ],
    python: `${loadSingle}

from panorai.ops.spherical import GaussianBlur

smoothed = GaussianBlur(sigma=1.0)(erp)
# Selecione Canny no Workbench para revisar limiares e suporte.
edges = smoothed`,
  },
  {
    id: "spherical-slic", index: "03", title: "Superpixels sobre a esfera",
    summary: "Divida a imagem em regiões coerentes com SLIC esférico.",
    detail: "Produz labels regionais para análise; não atribui classes semânticas.",
    audience: "Regiões · labels · segmentação clássica", inputCount: 1, inputLabel: "1 ERP RGB",
    outputs: ["labels", "fronteiras"], categories: ["analysis"], availability: "specialized", stability: "stable",
    route: "/single-view.html?experiment=slic-op", runtimeNote: "Pronto na bancada Single View", visual: "regions", featured: true,
    steps: [{ id: "slic", title: "SLIC esférico", publicApi: "panorai.image_processing.spherical_slic", componentPatterns: ["spherical_slic", "SLIC"], parents: ["input-a"] }],
    python: `${loadSingle}

from panorai.image_processing import spherical_slic

labels = spherical_slic(erp, n_segments=300, compactness=10.0)`,
  },
  {
    id: "spherical-features", index: "04", title: "Features e descritores esféricos",
    summary: "Detecte pontos distintivos sobre a ERP e preserve seus descritores.",
    detail: "Keypoints, resposta, escala e descritores ficam disponíveis para matching e pose.",
    audience: "Features · matching · localização", inputCount: 1, inputLabel: "1 ERP RGB",
    outputs: ["keypoints", "descritores"], categories: ["features"], availability: "specialized", stability: "experimental",
    route: "/single-view.html?experiment=dog-op", runtimeNote: "Detector DoG pronto; descritores dependem do fluxo escolhido", visual: "features",
    steps: [{ id: "dog", title: "DoG esférico", publicApi: "panorai.applications.SphericalDoGFeatures", componentPatterns: ["DoGDetector", "SphericalDog", "DoG"], parents: ["input-a"] }],
    python: `${loadSingle}

from panorai.applications import SphericalDoGFeatures

extractor = SphericalDogFeatures()
features = extractor(erp)`,
  },
  {
    id: "planar-mapping", index: "05", title: "Descoberta e medição de planos",
    summary: "Transforme linhas, pontos de fuga e altura conhecida em geometria revisável.",
    detail: "A experiência especializada separa orientação, posição, segmentação e publicação métrica.",
    audience: "Geometria 3D · ambientes internos · medição", inputCount: 1, inputLabel: "1 ERP + altura",
    outputs: ["planos", "range", "preview 3D"], categories: ["geometry", "3d"], availability: "specialized", stability: "experimental",
    route: "/known-height-planar.html", runtimeNote: "Fluxo especializado com revisão humana", visual: "planes", featured: true,
    steps: [
      { id: "discover", title: "Descoberta planar", publicApi: "panorai.reconstruction.planar.SphericalPlanarDiscovery", componentPatterns: ["SphericalPlanarDiscovery"], parents: ["input-a"] },
      { id: "map", title: "Mapa por altura conhecida", publicApi: "panorai.reconstruction.known_height_planar.KnownHeightPlanarMapping", componentPatterns: ["KnownHeightPlanarMapping"], parents: ["discover"] },
    ],
    python: `${loadSingle}

from panorai.reconstruction.planar import SphericalPlanarDiscovery
from panorai.reconstruction.known_height_planar import KnownHeightPlanarMapping

# Monte a cena tipada e revise as evidências antes de publicar geometria.
discovery = SphericalPlanarDiscovery()
mapping = KnownHeightPlanarMapping()`,
  },
  {
    id: "relative-pose", index: "06", title: "Pose relativa entre panoramas",
    summary: "Encontre correspondências e estime rotação e direção de translação.",
    detail: "O fluxo de duas vistas mantém matches, inliers e convenção de frame inspecionáveis.",
    audience: "Localização · SLAM · estrutura a partir de movimento", inputCount: 2, inputLabel: "2 ERPs",
    outputs: ["matches", "R", "t unitário"], categories: ["features", "3d"], availability: "specialized", stability: "experimental",
    route: "/two-view.html?experiment=relative-pose", runtimeNote: "Pronto com o par público conhecido", visual: "pose", featured: true,
    steps: [
      { id: "features", title: "Features do par", publicApi: "panorai.features.ExtractSphericalFeaturePair", componentPatterns: ["ExtractSphericalFeaturePair", "FeaturePair"], parents: ["input-a", "input-b"] },
      { id: "pose", title: "Pose relativa", publicApi: "panorai.estimators.estimate_relative_pose", componentPatterns: ["estimate_relative_pose", "RelativePose"], parents: ["features"] },
    ],
    python: `${loadPair}

from panorai.features import ExtractSphericalFeaturePair
from panorai.estimators import estimate_relative_pose

pair = ExtractSphericalFeaturePair()(erp_a, erp_b)
pose = estimate_relative_pose(pair)`,
  },
  {
    id: "spherical-stereo", index: "07", title: "Stereo denso e range radial",
    summary: "Use um par com pose conhecida para estimar distância na esfera.",
    detail: "Range, confiança, suporte e validade permanecem artefatos separados.",
    audience: "Depth · reconstrução · percepção 3D", inputCount: 2, inputLabel: "2 ERPs + pose",
    outputs: ["range radial", "confiança", "validade"], categories: ["depth", "3d"], availability: "specialized", stability: "experimental",
    route: "/two-view.html?experiment=spherical-stereo", runtimeNote: "Pronto com o par público e pose conhecida", visual: "stereo",
    steps: [
      { id: "range", title: "Stereo e range esférico", publicApi: "panorai.stereo.estimate_spherical_range", componentPatterns: ["estimate_spherical_range", "SphericalStereo"], parents: ["input-a", "input-b"], note: "A rotação e a translação métricas do par são inputs obrigatórios da operação." },
    ],
    python: `${loadPair}

from panorai.stereo import estimate_spherical_range

# Forneça a pose métrica no contrato X_b = R_b_from_a @ X_a + t_b_from_a.
range_result = estimate_spherical_range(
    erp_a,
    erp_b,
    rotation_b_from_a,
    translation_b_from_a,
)`,
  },
  {
    id: "spherical-metric-depth", index: "08", title: "Depth métrico convertido para esférico",
    summary: "Execute Metric3D portado para convoluções esféricas e obtenha range radial em metros.",
    detail: "O modelo e checkpoint permanecem externos. O carregamento exige Torch, aceite explícito dos termos upstream e uma ERP 2:1 compatível.",
    audience: "Deep learning · depth monocular · pesquisa", inputCount: 1, inputLabel: "1 ERP 2:1",
    outputs: ["range radial (m)", "relatório de port"], categories: ["depth", "models"], availability: "setup", stability: "experimental",
    runtimeNote: "Requer panorai[deep-learning-depth] + checkpoint externo", visual: "depth", featured: true,
    steps: [
      { id: "load", title: "Carregar Metric3D", publicApi: "panorai.experimental.deep_learning.load_spherical_metric3d_convnext_tiny_v1", componentPatterns: ["load_spherical_metric3d"], parents: [], note: "Download somente após aceite explícito dos termos upstream." },
      { id: "infer", title: "Inferência esférica", publicApi: "panorai.experimental.deep_learning.SphericalMetric3D", componentPatterns: ["SphericalMetric3D"], parents: ["input-a", "load"] },
    ],
    python: `import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from panorai.experimental.deep_learning import load_spherical_metric3d_convnext_tiny_v1

erp = pil_to_tensor(Image.open("panorama.jpg").convert("RGB")).float()[None]
loaded = load_spherical_metric3d_convnext_tiny_v1(
    accept_upstream_terms=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
radial_range_m = loaded.model(erp.to(next(loaded.model.model.parameters()).device))`,
  },
  {
    id: "semantic-instances", index: "09", title: "Propostas e instâncias com SAM",
    summary: "Use evidência visual para propor regiões e refine máscaras na esfera.",
    detail: "O pipeline preserva propostas, máscaras e suporte; modelos opcionais nunca são baixados silenciosamente.",
    audience: "Segmentação · modelos fundacionais · anotação", inputCount: 1, inputLabel: "1 ERP RGB",
    outputs: ["máscaras", "instâncias", "propostas"], categories: ["analysis", "models"], availability: "specialized", stability: "experimental",
    route: "/segmentation.html", runtimeNote: "CPU disponível; SAM requer runtime e checkpoint explícitos", visual: "segmentation",
    steps: [
      { id: "proposals", title: "Propostas esféricas", publicApi: "panorai.pipeline.CAMGuidedSphericalInstances", componentPatterns: ["CAMGuided", "Proposals"], parents: ["input-a"] },
      { id: "sam", title: "Refino SAM", publicApi: "panorai.pipeline.SAMMaskRefinement", componentPatterns: ["SAMMaskRefinement", "SAM"], parents: ["input-a", "proposals"] },
    ],
    python: `${loadSingle}

# O Workbench solicita aceite dos termos e configuração do modelo antes do SAM.
# As máscaras continuam separadas do suporte geométrico.`,
  },
  {
    id: "multiview-reconstruction", index: "10", title: "Reconstrução esférica multivista",
    summary: "Combine poses, tracks e observações em uma cena 3D inspecionável.",
    detail: "Para cenas registradas, o fluxo expõe câmeras, nuvem de pontos e métricas de ajuste.",
    audience: "SfM · mapeamento · pesquisa 3D", inputCount: 2, inputLabel: "2+ ERPs registradas",
    outputs: ["câmeras", "point cloud", "tracks"], categories: ["3d"], availability: "specialized", stability: "experimental",
    route: "/multi-view.html", runtimeNote: "Requer cena registrada e dependências do método escolhido", visual: "reconstruction",
    steps: [
      { id: "features", title: "Features e tracks", publicApi: "panorai.features.SphericalFeaturePipeline", componentPatterns: ["SphericalFeaturePipeline", "Feature"], parents: ["input-a", "input-b"] },
      { id: "reconstruct", title: "Reconstrução global", publicApi: "panorai.reconstruction.SphericalGlobalMapper", componentPatterns: ["SphericalGlobalMapper", "Reconstruction"], parents: ["features"] },
    ],
    python: `${loadPair}

from panorai.reconstruction import SphericalGlobalMapper

# Construa observações e poses tipadas antes da reconstrução.
reconstruction = SphericalGlobalMapper()`,
  },
];

export function getApplicationRecipe(id: string | null): ApplicationRecipe {
  return APPLICATION_RECIPES.find((recipe) => recipe.id === id) ?? APPLICATION_RECIPES[0];
}
