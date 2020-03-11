# auto_modeling

This is an integrated program helping analysts build various types of models, such as response model and risk model. The program includes the whole process of building models from data exploration, data preprocessing, modeling to model selection. To make the program work smoothly, please place the files to the corresponding folders(locations).

1. Data
- Contents:
  - Samples
  - Schema
  - Proba_schema: please make sure the value in the "Category" is the same as the column name of Y

2. Models
- Set up each scorecard's folder (it can be set/cleaned by python)
- Contents:
  - DataSplitter.pkl
  - Pipeline.pkl
  - Modeler.pkl 

3. Docs
- Set up each scorecard's folder (it can be set/cleaned by python)
- Content:
  - Results of variable preprocessing(.pdf/.csv)
  - Model result (Lift Chart)

=== Please do nothing to the following folders ===

4. sources
- The locations for the modules

5. Reports
- The location for temp reports


為確保程式可以順利運行，建模所需的檔案皆須依性質放入指定資料夾。

1. Data
存放內容: 
  - 樣本
  - 規格: 規格請依照Data_Schema.csv設計
  - proba_schema: 程式內部使用，請調整Category為Y的Field，使其與樣本Y欄位名稱一致


2. Models
- 依照評分卡設置各自的資料夾(可透過Python主程式設置/清空)
- 存放內容:
  - DataSplitter.pkl
  - Pipeline.pkl
  - Modeler.pkl 

3. Docs
- 依照評分卡設置各自的資料夾(可透過Python主程式設置/清空)
- 存放內容:
  - 各個preprocess階段的變數處理結果(pdf &csv)
  - 模型成效Lift Chart


=== 以下資料夾不需變動 ===

4. sources
- 自開發module的存放位置

5. Reports
- 程式內部產出的暫存位置
