using Microsoft.ML.Data;

namespace MLAPI.Model
{
    public class Prediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
