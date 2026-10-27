# BrowseComp-Verify website

Buildless HTML/CSS/JavaScript for the project page. No framework, npm install,
external fonts, analytics, API credentials, or inference service is needed.

Expected public address after deployment:
https://denebuniverse.github.io/BrowseComp-Verify/

## 本地预览

在开源仓库根目录（当前工作区的 `release/`）运行：

```bash
python3 -m http.server 8080 --bind 127.0.0.1 --directory website
```

浏览器打开 http://127.0.0.1:8080/ 。请使用 HTTP 预览，直接双击 HTML 的
`file://` 方式会被浏览器限制 JSON 加载。

## GitHub Pages 启用

这里的路径均以 **release 内容作为 GitHub 仓库根目录** 为前提。
不要把外层研究项目、`tmp/` 或原始结果一起发布。

1. 将审核后的 release 内容放入 `DenebUniverse/BrowseComp-Verify` 的 `main` 分支。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 选择 **GitHub Actions**。
3. 在 **Actions → Deploy project website → Run workflow** 手动执行首次部署。
4. 后续修改 `website/` 并推送到 main 会自动重新部署。
5. 等 workflow 成功后访问上方预期网址。尚未部署时该地址不代表网站已经上线。

工作流在 `../.github/workflows/pages.yml`，只上传 `website/`，不会上传数据集、
代码运行日志或检索资源。参考 [GitHub 官方文档](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
如果默认分支不是 main，请同步修改 workflow 的 `branches`。

## 内容与结果来源

- `index.html`：默认英文页面；`zh/index.html`：中文页面。
- `assets/styles.css`：桌面和手机布局，系统字体。
- `assets/main.js`：散点高亮、主图排序、strict/relaxed 切换、示例分步、BibTeX 复制。
- `data/results.json`：仅汇总指标，可独立下载，不含题目、候选或轨迹。

主图使用论文 Table 1；散点复现论文图中的报告值；prompt ablation 使用独立消融结果。
Human 只比较同一组 104 条，不并入主图。不包含 Best-of-N。
本网站没有进行新模型评测，也不需要付费 API。
