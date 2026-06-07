"""阶段 A 入口（占位）。

run.sh 调用：``python3 agent.py``。职责：装配各模块、启动 loop 主循环，
无论发生什么都保证 ``workspace/engine.py`` 存在并 ``exit 0``，失败信息写入
``results.log`` / ``output3.json``，绝不靠崩溃表达。

不变量：
- 始终 exit 0；任何异常都被捕获并降级为「这一轮/这次运行没收益」。
- 兜底链：最优正确候选 → 已验证 fallback(baseline) → 原始 fallback，三者至少其一落盘。
- 不在此处写业务逻辑，只做装配与顶层 try/finally；真正的循环在 loop。

接口/装配细节 TBD（Phase 2）。
"""
