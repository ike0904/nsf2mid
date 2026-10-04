・C:\Users\ike09\.claude\claude.md 初回起動時と更新あり時は必ず読むこと。
・docs/spec.md が仕様書。
・バージョンは src/nsf2mid.py の APP_VERSION で管理する。

---

## 作業記録

### v0.1.0 (2026-10-04) ← 最新
- プロジェクト新規作成（morokoshi の派生アプリ、プロンプト直接指示）
- Phase 1 実装: 6502 CPU コア・NSF ローダ（バンク切替対応）・APU 状態モデル・
  レジスタログ CSV / フレーム状態 CSV / トラッカー表示 TXT 出力
- 45 本の NSF で動作確認（全件エラーなし、10 秒分の処理が 0.6 秒以下）
- DQ2 の発音中 duty 変化、SMB の「トリガー後に音量 0→次フレームで発音」を確認し spec.md に記録
- テスト用 NSF は testdata/（著作物のため git 管理外）
