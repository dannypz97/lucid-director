import pytest

from lucid_director.core.registry import Registry, RegistryError


class Dummy:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_register_and_create() -> None:
    registry = Registry()
    registry.register("pipeline", "dummy", Dummy)

    obj = registry.create("pipeline", "dummy", 7)

    assert isinstance(obj, Dummy)
    assert obj.value == 7


def test_string_target_imported_lazily() -> None:
    registry = Registry()
    registry.register("display", "ordered", "collections:OrderedDict")

    from collections import OrderedDict

    assert registry.resolve("display", "ordered") is OrderedDict


def test_unknown_name_lists_known_names() -> None:
    registry = Registry()
    registry.register("frame_source", "stub", Dummy)

    with pytest.raises(RegistryError, match=r"'stbu'.*known: stub"):
        registry.resolve("frame_source", "stbu")


def test_duplicate_registration_rejected() -> None:
    registry = Registry()
    registry.register("llm", "scripted", Dummy)

    with pytest.raises(RegistryError, match="already registered"):
        registry.register("llm", "scripted", Dummy)


def test_unknown_kind_rejected() -> None:
    with pytest.raises(RegistryError, match="unknown kind"):
        Registry().available("audio")  # type: ignore[arg-type]


def test_missing_module_gives_readable_error() -> None:
    registry = Registry()
    registry.register("pipeline", "ghost", "lucid_director.nope:Ghost")

    with pytest.raises(RegistryError, match="cannot import"):
        registry.resolve("pipeline", "ghost")
