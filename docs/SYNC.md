# 同步与更新指南（nanochat 中文翻译）

本仓库是 `karpathy/nanochat` 的 fork，用于维护中文翻译（`README_zh.md`）与代码笔记。

- `origin`（你自己的 fork）：`https://github.com/weibifan/nanochat.git`
- `upstream`（上游官方）：`https://github.com/karpathy/nanochat.git`

## 工作流概览

每次上游 nanochat 更新后，按以下两步操作：

1. **同步上游代码** 到本地，再推送到自己的 fork。
2. **对比英文 README 的变化**，据此更新你的中文翻译 `README_zh.md`。

## 第 1 步：同步上游更新

```bash
# 先看上游有没有新提交
git fetch upstream

# 本地 master 落后于上游时，合并上游
git merge upstream/master

# 推送到你自己的 fork
git push origin master
```

### 关于冲突

- `README_zh.md` 是上游**不存在的文件**，合并时一般不会冲突。
- 只有当你**同时**改了上游也改过的同一处代码时才会冲突。若冲突，解决后：

```bash
git add <冲突文件>
git commit
git push origin master
```

## 第 2 步：根据英文 README 更新翻译

合并上游后，你的 `README_zh.md` 仍是旧版。用 diff 精确定位英文版改动的行，只更新对应段落，无需重读全文：

```bash
# 查看上游最新 README 相对本地 master 的改动
git diff master upstream/master -- README.md

# 或查看上游最近一次提交改了什么
git show upstream/master -- README.md
```

基于 diff 结果，更新 `README_zh.md` 中对应的中文段落，然后：

```bash
git add README_zh.md
git commit -m "Update Chinese translation to match upstream README"
git push origin master
```

## 常用辅助命令

```bash
# 查看远程配置
git remote -v

# 查看本地与上游差距（领先/落后几个提交）
git rev-list --left-right --count master...upstream/master

# 一键合并并推送
git fetch upstream && git merge upstream/master && git push origin master
```

## 推荐做法（一劳永逸，可选）

可以给 fork 配置 GitHub Action 实现**每天自动同步上游**：网页上进入 **Settings → Code and automation → Actions → General**，参考 `modded-nanogpt` 的做法添加定时 merge。这样 fork 永远保持最新，你只需专注更新中文翻译。