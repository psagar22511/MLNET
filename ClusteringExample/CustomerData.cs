using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ClusteringExample
{
    public class CustomerData
    {
        [LoadColumn(0)]
        public float Age;

        [LoadColumn(1)]
        public float AnnualIncome;
    }

    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}
