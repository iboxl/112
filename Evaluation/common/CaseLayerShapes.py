"""
Layer selection helpers shared by experiment scripts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, List, Optional

from Evaluation.common.EvalCommon import iter_model_layers


CASE_LAYERS_DETAILS = [
    {
        "id": "L1",
        "label": "Standard 3x3 mid-block conv",
        "source": "ResNet-18 Conv_8",
        "mechanism_role": "Balanced operand pressure, composite factorization. "
                          "Baseline behavior; sensitivity should be smooth.",
        "loopdim": {
            "R": 3, "S": 3, "P": 28, "Q": 28,
            "C": 128, "K": 128, "G": 1, "B": 1,
            "H": 28, "W": 28, "Stride": 1, "Padding": 1,
        },
    },
    {
        "id": "L2",
        "label": "Pointwise 1x1 deep",
        "source": "ResNet-18 Conv_17",
        "mechanism_role": "Weight-heavy, minimal spatial reuse, prime spatial "
                          "dims. Buffer pressure should hit weight residency "
                          "hard; bandwidth tests reload-trigger optimization.",
        "loopdim": {
            "R": 1, "S": 1, "P": 7, "Q": 7,
            "C": 256, "K": 512, "G": 1, "B": 1,
            "H": 7, "W": 7, "Stride": 1, "Padding": 0,
        },
    },
    {
        "id": "L3",
        "label": "Depthwise 3x3",
        "source": "MobileNet-v2 depthwise (C=K=G=144)",
        "mechanism_role": "G=C breaks weight reuse across groups; per-group "
                          "body small; partial-sum and mismatch decisions "
                          "dominate. Tests beta differential value.",
        "loopdim": {
            "R": 3, "S": 3, "P": 14, "Q": 14,
            "C": 1, "K": 1, "G": 144, "B": 1,
            "H": 14, "W": 14, "Stride": 1, "Padding": 1,
        },
    },
    {
        "id": "L4",
        "label": "Imbalanced 1x1 expansion",
        "source": "EfficientNet-B0 MBConv expansion (C=80, K=480)",
        "mechanism_role": "Channel-asymmetric, small spatial. Identified in "
                          "5.7 as fidelity worst case + ranking residual; "
                          "sensitivity probes the conditional regime.",
        "loopdim": {
            "R": 1, "S": 1, "P": 14, "Q": 14,
            "C": 80, "K": 480, "G": 1, "B": 1,
            "H": 14, "W": 14, "Stride": 1, "Padding": 0,
        },
    },
]


_LAYER_BY_ID = {layer["id"]: layer for layer in CASE_LAYERS_DETAILS}


def _annotate_model_layers(model_name: str, layers: Iterable[dict]) -> List[dict]:
    annotated = []
    for idx, layer in enumerate(layers):
        item = deepcopy(layer)
        item.setdefault("model", model_name)
        item["layer_index"] = idx
        item["layer_source"] = "model"
        item["layer_id"] = f"{model_name}:{item['layer']}"
        item["layer_aliases"] = [f"L{idx + 1}", f"layer{idx + 1}", f"Conv_{idx}"]
        annotated.append(item)
    return annotated


def _annotate_representative_layers(layers: Iterable[dict]) -> List[dict]:
    annotated = []
    for idx, layer in enumerate(layers):
        item = deepcopy(layer)
        item["model"] = "representative"
        item["layer"] = item["id"]
        item["layer_index"] = idx
        item["layer_source"] = "representative"
        item["layer_id"] = item["id"]
        item.setdefault("layer_type", "representative")
        item.setdefault("layer_family", "representative")
        annotated.append(item)
    return annotated


def _split_model_scope(selector: str, model_name: str):
    if ":" not in selector:
        return selector
    prefix, token = selector.split(":", 1)
    if prefix in {"idx", "index"}:
        return selector
    if prefix != model_name:
        return None
    return token


def _match_layer_token(layers: List[dict], token: str) -> List[dict]:
    if token in {"all", "*"}:
        return list(layers)
    if token in {"first", "head"}:
        return layers[:1]
    if token == "last":
        return layers[-1:] if layers else []

    if token.startswith("idx:") or token.startswith("index:"):
        idx = int(token.split(":", 1)[1])
        return [layers[idx]] if -len(layers) <= idx < len(layers) else []

    if token.isdigit():
        one_based = int(token)
        idx = one_based - 1
        return [layers[idx]] if 0 <= idx < len(layers) else []

    return [
        layer for layer in layers
        if (
            layer["layer"] == token or
            layer.get("layer_id") == token or
            token in layer.get("layer_aliases", [])
        )
    ]


def _select_by_tokens(model_name: str, layers: List[dict],
                      selectors: Optional[List[str]]) -> List[dict]:
    if not selectors:
        return list(layers)

    selected = []
    seen = set()
    unmatched = []
    for raw in selectors:
        token = _split_model_scope(raw, model_name)
        if token is None:
            continue
        matches = _match_layer_token(layers, token)
        if not matches:
            unmatched.append(raw)
            continue
        for layer in matches:
            key = layer["layer_id"]
            if key not in seen:
                selected.append(layer)
                seen.add(key)

    if unmatched:
        available = [layer["layer"] for layer in layers]
        raise ValueError(
            f"Unknown layer selector(s) for {model_name}: {unmatched}. "
            f"Use exact layer names, 1-based positions, idx:N, or "
            f"{model_name}:<selector>. Available layers: {available}"
        )
    return selected


def all_layer_ids() -> List[str]:
    return [layer["id"] for layer in CASE_LAYERS_DETAILS]


def layers_by_ids(ids: Optional[List[str]] = None) -> List[dict]:
    if not ids:
        return _annotate_representative_layers(CASE_LAYERS_DETAILS)
    requested = list(ids)
    unknown = [i for i in requested if i not in _LAYER_BY_ID]
    if unknown:
        raise ValueError(
            f"Unknown sensitivity layer IDs: {unknown}; "
            f"available: {all_layer_ids()}"
        )
    return _annotate_representative_layers(_LAYER_BY_ID[i] for i in requested)


def select_model_layers(model_name: str, layer_selectors: Optional[List[str]] = None,
                        max_layers: Optional[int] = None) -> List[dict]:
    """Return model layers, optionally filtered by CLI selectors.

    Selectors:
    - no selector: full model
    - exact parser layer name
    - generated aliases, e.g. L9 or Conv_8 for the ninth parsed layer
    - 1-based ordinal, e.g. 3
    - zero-based index, e.g. idx:2
    - model-scoped selector, e.g. resnet18:Conv_8 or resnet18:3

    max_layers preserves the old smoke-test behavior when no explicit layer
    selector is supplied. With selectors, it caps the selected subset.
    """
    layers = _annotate_model_layers(model_name, iter_model_layers(model_name))
    selected = _select_by_tokens(model_name, layers, layer_selectors)
    if max_layers is not None:
        selected = selected[:max_layers]
    return [deepcopy(layer) for layer in selected]


def layer_selection_config(layer_source: str = "model",
                           layer_selectors: Optional[List[str]] = None,
                           max_layers: Optional[int] = None) -> dict:
    return {
        "layer_source": layer_source,
        "layers": layer_selectors or "all",
        "max_layers": max_layers,
        "selector_semantics": (
            "full model by default; --layers accepts exact names, generated "
            "aliases (L9/Conv_8), 1-based positions, idx:N, or model:<selector>"
        ),
    }
