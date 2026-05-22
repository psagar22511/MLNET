using Microsoft.ML.Data;

namespace JobFinderML
{
    public class JobData
    {
        [LoadColumn(0)]
        public string Title { get; set; }
        [LoadColumn(1)]
        public string Description { get; set; }
        [LoadColumn(2)]
        public string Skills { get; set; }
        [LoadColumn(3)]
        public bool IsRelevant { get; set; } // Label (training)
    }
    public class JobPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
    }
}
