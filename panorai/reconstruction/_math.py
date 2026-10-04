"""Small SO(3) and spherical residual helpers."""

from __future__ import annotations

import math

import numpy as np


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def rotation_exp(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    cross = skew(vector)
    if angle < 1e-12:
        return np.eye(3) + cross + 0.5 * (cross @ cross)
    return (
        np.eye(3)
        + math.sin(angle) / angle * cross
        + (1.0 - math.cos(angle)) / angle**2 * (cross @ cross)
    )


def rotation_log(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )
    if angle < 1e-10:
        return 0.5 * vee
    if math.pi - angle < 1e-6:
        values, vectors = np.linalg.eigh((rotation + np.eye(3)) / 2.0)
        axis = vectors[:, int(np.argmax(values))]
        sign_index = int(np.argmax(np.abs(axis)))
        if axis[sign_index] < 0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * math.sin(angle)) * vee


def tangent_basis(direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=np.float64)
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(direction)))] = 1.0
    first = np.cross(direction, axis)
    first /= np.linalg.norm(first)
    return np.stack((first, np.cross(direction, first)), axis=1)


def spherical_log_residual(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    cosine = float(np.clip(np.dot(measured, predicted), -1.0, 1.0))
    angle = math.acos(cosine)
    tangent = predicted - cosine * measured
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-12:
        vector = tangent if angle < 1e-8 else angle * tangent_basis(measured)[:, 0]
    else:
        vector = angle / norm * tangent
    return tangent_basis(measured).T @ vector


def angular_error(measured: np.ndarray, predicted: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(measured, predicted), -1.0, 1.0)))
