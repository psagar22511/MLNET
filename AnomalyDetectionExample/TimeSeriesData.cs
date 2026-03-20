using Microsoft.ML.Data;

namespace AnomalyDetectionExample
{
    public class TimeSeriesData
    {
        public float Value { get; set; }
    }

    public class AnomalyPrediction
    {
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
