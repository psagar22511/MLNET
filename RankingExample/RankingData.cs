using Microsoft.ML.Data;

namespace RankingExample
{
    public class RankingData
    {
        public float Label { get; set; }

        [LoadColumn(1)]
        public float Feature1 { get; set; }

        [LoadColumn(2)]
        public float Feature2 { get; set; }

        //[KeyType(count: 100)]
        public uint GroupId { get; set; }
    }

    public class RankingPrediction
    {
        public float Score { get; set; }
    }
}
