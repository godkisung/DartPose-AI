class LLMFeedbackGenerator:
    def __init__(self, model_manager="ollama", model_name="llama3"):
        self.model_manager = model_manager
        self.model_name = model_name

    def generate_feedback(self, analysis_result_json):
        """
        Takes the rule engine results and generates natural language coaching feedback.
        """
        # Ex: "elbow_angle": 135, "issue": "elbow_too_bent", "speed": "fast"
        prompt = (f"다트 투구 분석 결과: {analysis_result_json}. "
                  "사용자가 팔꿈치를 더 펴도록 친절하게 조언해줘.")
        
        print(f"[{self.model_manager} - {self.model_name}] Prompt: {prompt}")
        
        # 1. Connect to Local LLM (e.g. via requests to http://localhost:11434/api/generate)
        # 2. Get Response
        
        mock_response = "다트를 던지실 때 팔꿈치가 조금 굽혀져 있어요! (135도) 릴리즈 순간 팔꿈치를 목표 방향으로 쭉(180도) 펴주시면 다트가 훨씬 일직선으로 날아갈 거예요. 파이팅!"
        return mock_response
