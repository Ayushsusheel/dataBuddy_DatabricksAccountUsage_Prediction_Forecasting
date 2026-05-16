run this file final_model_winner.py

python "C:\Users\ayush\Downloads\__Code__\ml_predictions\final_model_winner.py" `
  --input-csv "C:\Users\__Code__\ml_predictions\1yearOriginalData\OriginalAccountUsageData_1Year_2workspaceIDs.csv" `
  --ceo-actuals-csv "C:\Users\__Code__\ml_predictions\1stApr_30thApril_realData\OriginalAccountUsageData_1stApril2026_to30thApril2026_2worksapceIDs.csv" `
  --output-dir "C:\Users\__Code__\ml_predictions\outputs\workspace"



after running this we can see 

<output_dir>/
├── pipeline.log
├── processed/
├── eda/
├── benchmarks/
├── validation/
├── winner_artifacts/
├── report_assets/
├── reports/
├── serving/
└── metadata/




TERMINAL LOGS AFTER RUNNING THE FILE

(.venv) PS C:\Users\ml_predictions> python "C:\Users\__Code__\ml_predictions\final_model_winner.py" `
>>   --input-csv "C:\Users\__Code__\ml_predictions\1yearOriginalData\OriginalAccountUsageData_1Year_2workspaceIDs.csv" `
>>   --ceo-actuals-csv "C:\Users\ml_predictions\1stApr_30thApril_realData\OriginalAccountUsageData_1stApril2026_to30thApril2026_2worksapceIDs.csv" `
>>   --output-dir "C:\Users\__Code__\ml_predictions\outputs\workspace"
2026-05-16 17:10:05,582 | INFO | ===== PIPELINE START =====
2026-05-16 17:10:05,582 | INFO | Target = DAILY TOTAL COST
2026-05-16 17:10:05,584 | INFO | Validation = strict rolling backtests + optional April holdout validation
2026-05-16 17:10:05,584 | INFO | Deep learning intentionally omitted: ~365 daily target observations
2026-05-16 17:10:05,584 | INFO | Loading source CSV: C:\Users\__Code__\ml_predictions\1yearOriginalData\OriginalAccountUsageData_1Year_2workspaceIDs.csv
2026-05-16 17:10:05,669 | INFO | Filtered rows: 1,583
2026-05-16 17:10:05,669 | INFO | Workspace name: xyzz
2026-05-16 17:10:05,669 | INFO | Workspace ID used in outputs: abc123
2026-05-16 17:10:05,680 | INFO | Date range: 2025-04-01 to 2026-03-31
2026-05-16 17:10:05,768 | INFO | Saved processed table: product_raw.csv
2026-05-16 17:10:05,780 | INFO | Saved processed table: product_grouped.csv
2026-05-16 17:10:05,790 | INFO | Saved processed table: wide.csv
2026-05-16 17:10:05,795 | INFO | Saved processed table: daily_total.csv
2026-05-16 17:10:10,567 | INFO | EDA complete.
2026-05-16 17:10:10,571 | INFO | Backtest folds: [{'train_end': 240, 'test_start': 240, 'test_end': 270}, {'train_end': 270, 'test_start': 270, 'test_end': 300}, {'train_end': 300, 'test_start': 300, 'test_end': 330}, {'train_end': 330, 'test_start': 330, 'test_end': 360}]
2026-05-16 17:10:10,573 | INFO | === FOLD 1/4 | Train rows=240 | Test rows=30 ===
2026-05-16 17:10:10,573 | INFO | Running candidate: naive_full
2026-05-16 17:10:10,589 | INFO | Running candidate: seasonal_naive_full
2026-05-16 17:10:10,605 | INFO | Running candidate: autoarima_full
2026-05-16 17:10:12,213 | INFO | Running candidate: autoarima_recent270
2026-05-16 17:10:13,583 | INFO | Running candidate: autoarima_recent180
2026-05-16 17:10:14,003 | INFO | Running candidate: sarimax_full
2026-05-16 17:10:18,594 | INFO | Running candidate: sarimax_recent270
2026-05-16 17:10:23,430 | INFO | Running candidate: sarimax_recent180
2026-05-16 17:10:27,005 | INFO | Running candidate: prophet_full
17:10:27 - cmdstanpy - INFO - Chain [1] start processing
17:10:28 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:10:28,897 | INFO | Running candidate: prophet_recent270
17:10:29 - cmdstanpy - INFO - Chain [1] start processing
17:10:29 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:10:29,185 | INFO | Running candidate: prophet_recent180
17:10:29 - cmdstanpy - INFO - Chain [1] start processing
17:10:29 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:10:29,446 | INFO | Running candidate: xgb_full_raw
2026-05-16 17:10:31,324 | INFO | Running candidate: xgb_full_log
2026-05-16 17:10:32,975 | INFO | Running candidate: xgb_recent270_raw
2026-05-16 17:10:34,922 | INFO | Running candidate: xgb_recent270_log
2026-05-16 17:10:36,526 | INFO | Running candidate: xgb_recent180_raw
2026-05-16 17:10:37,833 | INFO | Running candidate: xgb_recent180_log
2026-05-16 17:10:39,172 | INFO | Running candidate: cat_full_raw
2026-05-16 17:10:42,513 | INFO | Running candidate: cat_full_log
2026-05-16 17:10:45,729 | INFO | Running candidate: cat_recent270_raw
2026-05-16 17:10:49,169 | INFO | Running candidate: cat_recent270_log
2026-05-16 17:10:52,546 | INFO | Running candidate: cat_recent180_raw
2026-05-16 17:10:59,112 | INFO | Running candidate: cat_recent180_log
2026-05-16 17:11:01,226 | INFO | === FOLD 2/4 | Train rows=270 | Test rows=30 ===
2026-05-16 17:11:01,226 | INFO | Running candidate: naive_full
2026-05-16 17:11:01,244 | INFO | Running candidate: seasonal_naive_full
2026-05-16 17:11:01,257 | INFO | Running candidate: autoarima_full
2026-05-16 17:11:06,012 | INFO | Running candidate: autoarima_recent270
2026-05-16 17:11:10,974 | INFO | Running candidate: autoarima_recent180
2026-05-16 17:11:11,539 | INFO | Running candidate: sarimax_full
2026-05-16 17:11:17,194 | INFO | Running candidate: sarimax_recent270
2026-05-16 17:11:22,855 | INFO | Running candidate: sarimax_recent180
2026-05-16 17:11:25,707 | INFO | Running candidate: prophet_full
17:11:25 - cmdstanpy - INFO - Chain [1] start processing
17:11:25 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:11:26,004 | INFO | Running candidate: prophet_recent270
17:11:26 - cmdstanpy - INFO - Chain [1] start processing
17:11:26 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:11:26,301 | INFO | Running candidate: prophet_recent180
17:11:26 - cmdstanpy - INFO - Chain [1] start processing
17:11:26 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:11:26,572 | INFO | Running candidate: xgb_full_raw
2026-05-16 17:11:28,052 | INFO | Running candidate: xgb_full_log
2026-05-16 17:11:29,747 | INFO | Running candidate: xgb_recent270_raw
2026-05-16 17:11:31,754 | INFO | Running candidate: xgb_recent270_log
2026-05-16 17:11:33,779 | INFO | Running candidate: xgb_recent180_raw
2026-05-16 17:11:35,550 | INFO | Running candidate: xgb_recent180_log
2026-05-16 17:11:37,451 | INFO | Running candidate: cat_full_raw
2026-05-16 17:11:41,581 | INFO | Running candidate: cat_full_log
2026-05-16 17:11:45,309 | INFO | Running candidate: cat_recent270_raw
2026-05-16 17:11:49,126 | INFO | Running candidate: cat_recent270_log
2026-05-16 17:11:52,892 | INFO | Running candidate: cat_recent180_raw
2026-05-16 17:11:55,678 | INFO | Running candidate: cat_recent180_log
2026-05-16 17:11:58,415 | INFO | === FOLD 3/4 | Train rows=300 | Test rows=30 ===
2026-05-16 17:11:58,415 | INFO | Running candidate: naive_full
2026-05-16 17:11:58,430 | INFO | Running candidate: seasonal_naive_full
2026-05-16 17:11:58,444 | INFO | Running candidate: autoarima_full
2026-05-16 17:12:01,789 | INFO | Running candidate: autoarima_recent270
2026-05-16 17:12:06,648 | INFO | Running candidate: autoarima_recent180
2026-05-16 17:12:07,059 | INFO | Running candidate: sarimax_full
2026-05-16 17:12:12,876 | INFO | Running candidate: sarimax_recent270
2026-05-16 17:12:18,509 | INFO | Running candidate: sarimax_recent180
2026-05-16 17:12:22,817 | INFO | Running candidate: prophet_full
17:12:22 - cmdstanpy - INFO - Chain [1] start processing
17:12:22 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:12:23,097 | INFO | Running candidate: prophet_recent270
17:12:23 - cmdstanpy - INFO - Chain [1] start processing
17:12:23 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:12:23,391 | INFO | Running candidate: prophet_recent180
17:12:23 - cmdstanpy - INFO - Chain [1] start processing
17:12:23 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:12:23,663 | INFO | Running candidate: xgb_full_raw
2026-05-16 17:12:25,179 | INFO | Running candidate: xgb_full_log
2026-05-16 17:12:26,954 | INFO | Running candidate: xgb_recent270_raw
2026-05-16 17:12:29,420 | INFO | Running candidate: xgb_recent270_log
2026-05-16 17:12:31,877 | INFO | Running candidate: xgb_recent180_raw
2026-05-16 17:12:34,127 | INFO | Running candidate: xgb_recent180_log
2026-05-16 17:12:36,667 | INFO | Running candidate: cat_full_raw
2026-05-16 17:12:43,781 | INFO | Running candidate: cat_full_log
2026-05-16 17:12:51,698 | INFO | Running candidate: cat_recent270_raw
2026-05-16 17:13:01,113 | INFO | Running candidate: cat_recent270_log
2026-05-16 17:13:10,204 | INFO | Running candidate: cat_recent180_raw
2026-05-16 17:13:17,484 | INFO | Running candidate: cat_recent180_log
2026-05-16 17:13:25,052 | INFO | === FOLD 4/4 | Train rows=330 | Test rows=30 ===
2026-05-16 17:13:25,052 | INFO | Running candidate: naive_full
2026-05-16 17:13:25,062 | INFO | Running candidate: seasonal_naive_full
2026-05-16 17:13:25,075 | INFO | Running candidate: autoarima_full
2026-05-16 17:13:29,690 | INFO | Running candidate: autoarima_recent270
2026-05-16 17:13:30,098 | INFO | Running candidate: autoarima_recent180
2026-05-16 17:13:33,982 | INFO | Running candidate: sarimax_full
2026-05-16 17:13:40,283 | INFO | Running candidate: sarimax_recent270
2026-05-16 17:13:45,348 | INFO | Running candidate: sarimax_recent180
2026-05-16 17:13:48,797 | INFO | Running candidate: prophet_full
17:13:48 - cmdstanpy - INFO - Chain [1] start processing
17:13:48 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:13:49,081 | INFO | Running candidate: prophet_recent270
17:13:49 - cmdstanpy - INFO - Chain [1] start processing
17:13:49 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:13:49,386 | INFO | Running candidate: prophet_recent180
17:13:49 - cmdstanpy - INFO - Chain [1] start processing
17:13:49 - cmdstanpy - INFO - Chain [1] done processing
2026-05-16 17:13:49,649 | INFO | Running candidate: xgb_full_raw
2026-05-16 17:13:51,921 | INFO | Running candidate: xgb_full_log
2026-05-16 17:13:54,541 | INFO | Running candidate: xgb_recent270_raw
2026-05-16 17:13:56,628 | INFO | Running candidate: xgb_recent270_log
2026-05-16 17:13:59,336 | INFO | Running candidate: xgb_recent180_raw
2026-05-16 17:14:01,102 | INFO | Running candidate: xgb_recent180_log
2026-05-16 17:14:02,996 | INFO | Running candidate: cat_full_raw
2026-05-16 17:14:08,331 | INFO | Running candidate: cat_full_log
2026-05-16 17:14:14,523 | INFO | Running candidate: cat_recent270_raw
2026-05-16 17:14:18,841 | INFO | Running candidate: cat_recent270_log
2026-05-16 17:14:22,892 | INFO | Running candidate: cat_recent180_raw
2026-05-16 17:14:25,818 | INFO | Running candidate: cat_recent180_log
2026-05-16 17:14:28,954 | INFO | Strict backtest artifacts saved.
2026-05-16 17:14:28,954 | INFO | Loading holdout actuals for exact April validation scoring...
2026-05-16 17:14:28,958 | INFO | Loading source CSV: C:\Users\__Code__\ml_predictions\1stApr_30thApril_realData\OriginalAccountUsageData_1stApril2026_to30thApril2026_2worksapceIDs.csv
2026-05-16 17:14:28,990 | INFO | Filtered rows: 196
2026-05-16 17:14:28,991 | INFO | Workspace name: abc
2026-05-16 17:14:28,992 | INFO | Workspace ID used in outputs: abc123
2026-05-16 17:14:28,992 | INFO | Date range: 2026-04-01 to 2026-04-30
2026-05-16 17:14:29,062 | INFO | Top candidates to score on holdout validation: ['xgb_full_raw', 'xgb_recent270_log', 'cat_full_raw', 'xgb_full_log', 'xgb_recent180_log']
2026-05-16 17:14:29,063 | INFO | Scoring candidate on holdout validation: xgb_full_raw
2026-05-16 17:14:32,360 | INFO | Scoring candidate on holdout validation: xgb_recent270_log
2026-05-16 17:14:36,429 | INFO | Scoring candidate on holdout validation: cat_full_raw
2026-05-16 17:14:48,151 | INFO | Scoring candidate on holdout validation: xgb_full_log
2026-05-16 17:14:50,423 | INFO | Scoring candidate on holdout validation: xgb_recent180_log
2026-05-16 17:14:52,615 | INFO | Final holdout winner: xgb_full_raw
2026-05-16 17:14:52,645 | INFO | Holdout validation scoring complete.
2026-05-16 17:14:56,100 | INFO | Final HTML report written to: C:\Users\__Code__\ml_predictions\outputs\workspace_abc\reports\forecast_report.html
2026-05-16 17:14:56,100 | INFO | ===== PIPELINE COMPLETE =====
(.venv) PS C:\Users\__Code__\ml_predictions> 
