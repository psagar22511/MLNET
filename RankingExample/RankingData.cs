namespace RankingExample
{
    public class RankingData
    {
        public float Label { get; set; }
        public float Feature1 { get; set; }
        public float Feature2 { get; set; }
        public int GroupId { get; set; }
    }

    public class RankingPrediction
    {
        public float Score { get; set; }
    }
}
