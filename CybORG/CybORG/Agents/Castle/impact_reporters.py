from CybORG.Shared.Actions import Impact
from CybORG.Shared.Enums import TrinaryEnum
from typing import Tuple, Dict

# KEYS
AUTH_KEY = "Auth"
DATABASE_KEY = "Database"
FRONT_KEY = "Front"

IMPACT_TYPE_KEY = "impact_type"
AVAIL_DROP_KEY = "avail_drop"
INTEGRITY_DROP_KEY = "integrity_drop"
CONF_DROP_KEY = "conf_drop"

IMPACT_NONE = 0
IMPACT_AUTH = 1
IMPACT_DATABASE = 2
IMPACT_FRONT = 3


# CHECKS
def _check_is_impact(red_action) -> bool:
    return type(red_action) is Impact


def _check_is_success(red_success) -> bool:
    return red_success == TrinaryEnum.TRUE


def check_is_successful_impact(red_action, red_success) -> bool:
    return _check_is_impact(red_action) and _check_is_success(red_success)


def _is_auth(hostname: str) -> bool:
    return AUTH_KEY in hostname


def _is_db(hostname: str) -> bool:
    return "Database" in hostname


def _is_front(hostname: str) -> bool:
    return "Front" in hostname


class ImpactReporter:
    def __call__(self, red_action, red_success) -> Tuple[int, Dict[str, int]]:
        if check_is_successful_impact(red_action, red_success):
            hostname = red_action.hostname
            if _is_auth(hostname):
                return IMPACT_AUTH, {AUTH_KEY: IMPACT_AUTH}
            elif _is_db(hostname):
                return IMPACT_DATABASE, {DATABASE_KEY: IMPACT_DATABASE}
            elif _is_front(hostname):
                return IMPACT_FRONT, {FRONT_KEY: IMPACT_FRONT}
            else:
                pass
        return IMPACT_NONE, {}
