# 上传到 GitHub 的步骤

## 1. 在 GitHub 上创建新仓库

1. 登录 [GitHub](https://github.com)，点击右上角 **+** → **New repository**
2. 填写仓库名（例如 `Time-LLM` 或 `Time-LLM-Iron`）
3. 选择 **Public**，**不要**勾选 "Add a README file"
4. 点击 **Create repository**

## 2. 在本地添加你的仓库并推送

创建好仓库后，GitHub 会显示仓库地址，例如：`https://github.com/你的用户名/仓库名.git`

在项目目录下执行（把下面的地址换成你的仓库地址）：

```bash
# 添加你的 GitHub 仓库为新的远程（保留原作者的 origin 可选）
git remote add mygithub https://github.com/你的用户名/仓库名.git

# 提交当前修改（包括 .gitignore）
git add .gitignore
git add -A
git status   # 确认要提交的文件
git commit -m "Add project with Iron experiments and .gitignore"

# 推送到你的仓库
git push -u mygithub main
```

如果希望**只**推送到你的仓库（不再保留原作者的 origin）：

```bash
git remote set-url origin https://github.com/你的用户名/仓库名.git
git push -u origin main
```

## 3. 若推送时要求认证

- **HTTPS**：会提示输入 GitHub 用户名和密码；密码需使用 [Personal Access Token](https://github.com/settings/tokens) 而非登录密码。
- **SSH**：如已配置 SSH key，可把远程地址改为 `git@github.com:你的用户名/仓库名.git` 再推送。

## 说明

- 当前远程 `origin` 指向原项目：`https://github.com/KimMeen/Time-LLM.git`
- `.gitignore` 已添加，会忽略 `__pycache__/`、`checkpoints/` 等，避免提交缓存和大型模型文件。
- 若希望把本地结果文件（如 `arima_eval_result.txt`）也排除，可取消 `.gitignore` 里对应行的注释。
