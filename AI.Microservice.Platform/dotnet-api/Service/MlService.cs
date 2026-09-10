using dotnet_api.Model;

namespace dotnet_api.Service
{
    public class MlService
    {
        private readonly HttpClient _httpClient;

        public MlService(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<string> GetPrediction(PredictionRequest request)
        {
            var response = await _httpClient.PostAsJsonAsync("http://127.0.0.1:8000/docs#/default/get_prediction_predict_post", request);
            return await response.Content.ReadAsStringAsync();
        }
    }
}
