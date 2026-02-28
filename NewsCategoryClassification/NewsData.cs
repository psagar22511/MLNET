using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace NewsCategoryClassification
{
    public class NewsData
    {
        [LoadColumn(0)]
        public string Text { get; set; }
        [LoadColumn(1)]
        public string Label { get; set; }
    }

    public class NewsPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }
}
