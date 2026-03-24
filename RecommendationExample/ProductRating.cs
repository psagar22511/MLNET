using Microsoft.ML.Data;

namespace RecommendationExample
{
    public class ProductRating
    {
        [KeyType(count: 100)]
        public uint userId;

        [KeyType(count: 100)]
        public uint productId;

        public float Label;
    }

    public class ProductPrediction
    {
        public float Score;
    }
}
