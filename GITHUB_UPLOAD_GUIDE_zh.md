# GitHub 上传指南

在 Windows PowerShell 打开整理后的项目目录。建议 GitHub 仓库名为 `New-Framework-of-Robust-Image-Encryption`。首次上传时，在实验室账号或组织下创建同名空仓库，然后核对暂存清单。

```powershell
Set-Location -LiteralPath 'C:\sys_github\New-Framework-of-Robust-Image-Encryption'
git init -b main
git add -- .gitignore README.md requirements.txt GITHUB_UPLOAD_GUIDE_zh.md assets docs train_code
git --no-pager diff --cached --stat
git --no-pager diff --cached --name-only
```

清单应包含源码、论文标题图和项目文档。数据集、权重、实验输出和完整论文保存在 Git 之外。

如果空仓库建在 `usst-micslab` 组织下，用以下命令提交并推送。仓库归属不同时，把远端地址换成实际的 HTTPS 地址。

```powershell
git commit -m "Prepare paper code for release"
git remote add origin https://github.com/usst-micslab/New-Framework-of-Robust-Image-Encryption.git
git remote -v
git push -u origin main
```

如果此目录后来已关联远端，先用 `git remote -v` 核对地址。日常更新依次执行暂存、检查、提交和 `git push`。上传后刷新 GitHub，查看 README 标题图、源码目录和论文链接。
