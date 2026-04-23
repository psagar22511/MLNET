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


    /* Item Size    Weight Hardness
    Apple	2	1	1
    Banana	3	1	1
    Toy Car 4   3	3
    Ball	3	2	2 */
    public class ClusterItemData2
    {
        [LoadColumn(0)]
        public float Size;

        [LoadColumn(1)]
        public float Weight;

        [LoadColumn(2)]
        public float Hardness;

        public string FruitName;
    }
    public class ClusterPrediction2
    {
        [ColumnName("PredictedLabel")]
        public uint ClusterId;

        public float[] Distances;
    }
}
