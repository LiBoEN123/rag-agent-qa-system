# structured-table.pdf

1,933-byte、单页数字文本 PDF，专用于真实 Docling 表格解析测试。内容为：

```text
Paragraph A: registration requirements.
Item         | Condition     | Material
Registration | Age 18+       | Identity card
Renewal      | Active status | Application
Paragraph B: contact the service desk.
```

使用一次性的 ReportLab 4.4.3 生成，不需要在测试或生产依赖中声明 ReportLab：
`SimpleDocTemplate(..., invariant=1)`；段落用默认 Normal 样式，表格前后
`Spacer(1, 24)`；`Table` 三列各 150 pt、三行各 32 pt；黑色 1 pt GRID，
首行浅灰背景与 Helvetica-Bold，12 pt 字体，单元格垂直居中。

测试直接读取这个固定小文件，不自行实现 PDF writer/layout engine。
