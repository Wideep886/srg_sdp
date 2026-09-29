# Validation record

The current revision was checked on **2026-09-29**, using macOS arm64,
Python 3.9.6, and an existing SDPA-GMP executable. The compact
[validation report](../results/release_checks/current_validation.json) records
the outcomes and SHA-256 hashes of the tested code, tests, and saved certificates.

| Check | Result |
|---|---|
| Unit and regression tests | **56 passed**, including rejection of ambiguous JSON, invalid coefficients, changed inputs, stale success reports, failed processes, and invalid timeouts. |
| `(16,6,2,2)`, three points, degree five | Complete search and exact verification gave bound **16**; the independent checker also passed. |
| `(8,6,4,6)`, four points, degree two | Complete search and exact verification gave bound **8**; the independent checker also passed. This case includes dependent reference configurations. |
| Saved `(550,162,75,36)`, six points, degree five | All three required verification stages passed again, giving bound **506**, with no new solver run. |
| All four saved certificates | Parameter-to-model linkage, strict JSON/coefficient parsing, and unchanged file contents checked. |
| Paths and documentation | Printed commands ran in zsh with spaces and apostrophes in project/output paths. Bilingual commands and local document links agree. |
| Bundled sources | Both archive hashes and archive-content checks passed. Third-party files were unchanged. |

Run the small test suite from the project root:

```sh
python3 -B -m unittest discover -s extras/tests -v
```

The other three six-point certificates were **not reverified for feasibility**
in this revision. Their original files and historical reports are unchanged.
The optional independent checker was tested on the two small examples, not on
a six-point certificate. No full experimental table was searched again, and
the solver was not rebuilt.

Earlier reports in `results/release_checks/` record the six-point model
comparison: regenerated coefficients, shifts, and SDPA input for
`(550,162,75,36)` matched the archived files byte for byte. Those reports identify
the versions tested at that time. An earlier solver build passed GMP tests and
the upstream SDP example.

File hashes establish content integrity. Tests check program behavior. Exact
certificate verification establishes the bound under the stated mathematical
formulation. These checks are not a proof-assistant formalization.

## 中文說明

本版於 **2026-09-29** 使用 macOS arm64、Python 3.9.6 與既有 SDPA-GMP
檢查。[驗證摘要](../results/release_checks/current_validation.json) 保留結果
及受測程式、測試與既有證書的檔案雜湊。

- **56 項測試全部通過**，涵蓋歧義輸入、錯誤係數、檔案被改動、過期成功
  紀錄、子程序失敗及不合法的逾時設定。
- `(16,6,2,2)` 的三點、五次完整搜尋與精確驗證得到上界 **16**；
  `(8,6,4,6)` 的四點、二次得到 **8**。兩者均通過獨立檢查，沒有錯誤
  排除已知存在的圖；後者包含相依參考配置。
- 既有 `(550,162,75,36)` 六點、五次證書重新通過三階段精確驗證，
  上界仍為 **506**，未重新執行求解器。
- 四份證書都通過參數對應、嚴格讀檔及檔案完整性檢查。含空白與單引號
  路徑的操作、中英文指令、文件連結及第三方來源壓縮檔檢查也通過。

其餘三份六點證書本次未重驗可行性；獨立檢查也未用於六點案例。四份原始
證書與第三方檔案均未改動。本次未重跑整表或重新編譯求解器。較早的模型
比對與驗證報告仍保留，並依其當時的程式版本解讀。

檔案雜湊用於核對內容，軟體測試用於檢查程式行為；球面上界由精確證書
驗證建立。這是精確算術的電腦輔助驗證，並非 proof assistant 的形式化證明。
