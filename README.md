# gym-v

## 开发前可以先安装一下pre-commit (uv run pre-commit install)

## 阶段一

我们先给一些已经有的env集成进来，新增的env需要：
- 和原仓库的env的游戏行为一致
- 代码也尽量和原仓库保持一致（方便review）
- 如果原仓库是纯文本env，集成进来需要新增出图函数并且可以注意一下game description部分不要再包含原来的与纯文本caption / maze相关的内容
- 如果是纯文本env，在实现时候以注释形式写一下原本文本env的一个qa的case，新的game description还需要注意需要与文本env预期的回答format和现在的render的图对的上（可参考下rg在gym-v得写法
