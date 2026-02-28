using Microsoft.ML.Data;

namespace MLAPI.Model
{
    public class HouseData
    {
        [LoadColumn(0)]
        public float Size { get; set; }
        [LoadColumn(1)]
        public float Bedrooms { get; set; }
        [LoadColumn(2)]
        public float Age { get; set; }
        [LoadColumn(3)]
        public float Label { get; set; }  // Label
    }

    public class HousePrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}
