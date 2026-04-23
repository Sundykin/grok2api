"""Virtual image-model aliases — dynamic routing based on request content.

An alias exposes one public model name that the chat endpoint rewrites to a
real model at dispatch time:

  - requests with any ``type=image_url`` block → ``image_to_image`` target
  - text-only requests                         → ``text_to_image`` target

At least one leg must be configured. When a request hits the unconfigured
leg, the endpoint returns a validation error — the alias is never a silent
fallback to the other direction.
"""

from dataclasses import dataclass
from typing import Any

from app.platform.logging.logger import logger

from . import registry as _registry
from .spec import ModelSpec


@dataclass(slots=True, frozen=True)
class ImageAlias:
    name: str
    public_name: str
    text_to_image: str | None
    image_to_image: str | None


def _coerce_bool(val: Any, default: bool = True) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _validate_leg(
    alias_name: str,
    target: str | None,
    cap_label: str,
    predicate,
) -> str | None:
    if not target:
        return None
    spec = _registry.get(target)
    if spec is None or not spec.enabled:
        logger.warning(
            "Model alias {!r} → {!r} ignored: target not registered or disabled",
            alias_name, target,
        )
        return None
    if not predicate(spec):
        logger.warning(
            "Model alias {!r} → {!r} ignored: target lacks {} capability",
            alias_name, target, cap_label,
        )
        return None
    return target


def _parse(name: str, raw: Any) -> ImageAlias | None:
    if not isinstance(raw, dict):
        logger.warning("Model alias {!r} ignored: expected table", name)
        return None
    if not _coerce_bool(raw.get("enabled", True), default=True):
        return None

    t2i = _validate_leg(name, raw.get("text_to_image"), "IMAGE", ModelSpec.is_image)
    i2i = _validate_leg(name, raw.get("image_to_image"), "IMAGE_EDIT", ModelSpec.is_image_edit)
    if t2i is None and i2i is None:
        logger.warning(
            "Model alias {!r} ignored: no valid text_to_image or image_to_image leg",
            name,
        )
        return None

    public_name = raw.get("public_name") or name
    return ImageAlias(
        name=name,
        public_name=str(public_name),
        text_to_image=t2i,
        image_to_image=i2i,
    )


# Parsed-aliases cache keyed by the section dict's object identity. Each
# config reload builds a fresh nested dict, so identity change is a reliable
# invalidation signal and also suppresses repeated warning logs for the same
# broken config.
_cache: tuple[int, dict[str, ImageAlias]] | None = None


def load_aliases(cfg) -> dict[str, ImageAlias]:
    """Parse the current config snapshot's ``models.alias`` section.

    Cheap to call per request — parses once per config reload.
    """
    global _cache

    section = cfg.get("models.alias")
    if not isinstance(section, dict):
        return {}

    section_id = id(section)
    if _cache is not None and _cache[0] == section_id:
        return _cache[1]

    result: dict[str, ImageAlias] = {}
    for name, raw in section.items():
        if _registry.get(name) is not None:
            logger.warning(
                "Model alias {!r} ignored: name collides with a real model",
                name,
            )
            continue
        alias = _parse(name, raw)
        if alias is not None:
            result[name] = alias

    _cache = (section_id, result)
    return result


def lookup(cfg, model_name: str) -> ImageAlias | None:
    """Return the alias entry for *model_name*, or None if it is not an alias."""
    return load_aliases(cfg).get(model_name)


def target_for(alias: ImageAlias, *, has_image: bool) -> str | None:
    """Return the real model name for the requested direction, or None."""
    return alias.image_to_image if has_image else alias.text_to_image


__all__ = ["ImageAlias", "load_aliases", "lookup", "target_for"]
