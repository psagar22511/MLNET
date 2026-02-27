using Microsoft.ML.Data;

namespace RegressionHousePricePredict
{
    public class HouseData
    {
        [LoadColumn(0)]
        public float Size;
        [LoadColumn(1)]
        public float Bedrooms;
        [LoadColumn(2)]
        public float Age;
        [LoadColumn(3)]
        public float Label;   // Label
    }

    public class HousePrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice;
    }
}
