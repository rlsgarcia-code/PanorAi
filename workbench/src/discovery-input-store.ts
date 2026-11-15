import type { DiscoveryInput, RecipeDraft } from "./discovery-model.ts";

const DATABASE = "panorai-workbench-inputs-v1";
const STORE = "assets";
const DRAFT_PREFIX = "panorai-workbench.recipe.";

export interface StoredAsset {
  descriptor: DiscoveryInput;
  blob: Blob;
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) {
        request.result.createObjectStore(STORE, { keyPath: "descriptor.id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB unavailable"));
  });
}

export async function storeDiscoveryAsset(
  descriptor: DiscoveryInput,
  blob: Blob,
): Promise<void> {
  const database = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(STORE, "readwrite");
    transaction.objectStore(STORE).put({ descriptor, blob } satisfies StoredAsset);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("Input storage failed"));
  });
  database.close();
}

export async function loadDiscoveryAsset(id: string): Promise<StoredAsset | null> {
  const database = await openDatabase();
  const value = await new Promise<StoredAsset | undefined>((resolve, reject) => {
    const transaction = database.transaction(STORE, "readonly");
    const request = transaction.objectStore(STORE).get(id);
    request.onsuccess = () => resolve(request.result as StoredAsset | undefined);
    request.onerror = () => reject(request.error ?? new Error("Input lookup failed"));
  });
  database.close();
  return value ?? null;
}

export function storeRecipeDraft(draft: RecipeDraft): void {
  sessionStorage.setItem(`${DRAFT_PREFIX}${draft.id}`, JSON.stringify(draft));
}

export function loadRecipeDraft(id: string): RecipeDraft | null {
  const raw = sessionStorage.getItem(`${DRAFT_PREFIX}${id}`);
  if (!raw) return null;
  const value = JSON.parse(raw) as RecipeDraft;
  return value.schema === "panorai-workbench-recipe-draft/v1" ? value : null;
}
