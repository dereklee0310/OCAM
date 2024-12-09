from yacs.config import CfgNode as CN

_C = CN()

_C.DISCORD = CN()
_C.DISCORD.WEBHOOK_URL = ""
_C.DISCORD.TEST_URL = ""
_C.DISCORD.USERNAME = ""
_C.DISCORD.AVATAR_URL = ""
_C.DISCORD.FAIL_MSG = "我掛了 :sob:"

_C.DISCORD.EMBED = CN()
_C.DISCORD.EMBED.TITLE = "123"
_C.DISCORD.EMBED.COLOR = 2309611

_C.DISCORD.EMBED.FOOTER = CN()
_C.DISCORD.EMBED.FOOTER.TEXT = ""
_C.DISCORD.EMBED.FOOTER.ICON_URL = ""

_C.DOUYIN = CN()
_C.DOUYIN.SID = "" # fuck you yacs :)

_C.DELAY = 5


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
