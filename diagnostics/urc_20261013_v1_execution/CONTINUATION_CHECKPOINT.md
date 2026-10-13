# URC-v1 execution checkpoint (working notes; not evidence)

- 09:26:04 +08:00: authorization recorded (EXECUTION_AUTHORIZATION.md).
- Pre-launch check PASS (prelaunch_check.md, prelaunch_verify.out.txt).
- 09:33:09–09:34:38: the ONE scientific invocation ran (launch/). Exit 0. 1 invocation, 1 start, R-FIT
  COMPLETED_FIT_FAIL, 3,000 updates, arm 87.531 s, no guard action, empty stderr. THE ALLOWANCE IS USED: never relaunch.
- 09:35:11: urc_analyze.py → analysis/ (analysis.json 728f7934…; sealed run root + mirror, 15 entries).
- Independent verification (independent_verification.json) and forward-only replay (replay_final_state.json): all agree,
  replay bit-identical. MATCHED (35 conditions), strict COMPLETED_FIT_FAIL, EP PASS, δRSE −0.0484742732858518,
  δcos +0.02948609068752228.
- REPORT.md written and read back; corrections made: 35 not 36 conditions; curve not monotone at the end; 2 ulp.
- EVIDENCE_INVENTORY_SHA256.txt written last. TASK COMPLETE; nothing committed; no further run authorized.
