# 环境噪声钉死 (U_env pinning / common random numbers)
#
# 目的：让假设 A3（环境可重放）在实证上成立。任何随时间/区域/终端宽度变化的输出
# 都会成为不可控的外生噪声，破坏前缀重放的可比性。
#
# 用法：source 本文件，或由 harness/appserver.py 的 PINNED_ENV 以等价方式注入。

# 排序与文本比较必须与区域设置无关
export LC_ALL=C
export LANG=C

# 时间戳与相对时间必须稳定
export TZ=UTC

# 关闭一切颜色/控制字符
export NO_COLOR=1
export CLICOLOR=0
export CLICOLOR_FORCE=0
export TERM=dumb

# 终端宽度会改变很多工具的换行/截断行为
export COLUMNS=200
export LINES=50

# Python 侧的哈希随机化会改变 dict/set 遍历顺序
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

# git 不读系统/全局配置，不弹交互
export GIT_CONFIG_NOSYSTEM=1
export GIT_TERMINAL_PROMPT=0
export GIT_PAGER=cat
export PAGER=cat

# ripgrep 不读用户配置（否则 --smart-case 等会改变命中集）
export RIPGREP_CONFIG_PATH=/dev/null

# 关闭各类 CLI 的匿名统计与更新检查（会引入网络时延与非确定性输出）
export DO_NOT_TRACK=1
export CI=1
