import uuid

from hivemind_exp.name_utils import get_name_from_uuid, search_uuid_for_name

TEST_UUID = "00000000-0000-0000-0000-000000000000"
TEST_UUIDS = [TEST_UUID[:-1] + str(i) for i in range(10)]

def test_get_name_from_uuid():
    id0 = uuid.UUID(TEST_UUIDS[0])
    id1 = uuid.UUID(TEST_UUIDS[1])
    id2 = uuid.UUID(TEST_UUIDS[2])
    assert get_name_from_uuid(str(id0)) == "scavenging owl"
    assert get_name_from_uuid(str(id1)) == "stocky ladybug"
    assert get_name_from_uuid(str(id2)) == "playful caribou"
    assert get_name_from_uuid(str(id2), True) == "playful_caribou"


def test_search_uuid_for_name():
    assert search_uuid_for_name(TEST_UUIDS, "none") is None
    assert search_uuid_for_name(TEST_UUIDS, "stocky ladybug") == TEST_UUIDS[1]
