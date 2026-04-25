import json
import os
import re
import time
from typing import Dict
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from tavily import TavilyClient

load_dotenv()


class LinkedInPostGenerator:
    VALID_TONES = {"professional", "casual", "thought-leader"}
    VALID_TYPES = {"story", "hot-take", "announcement", "lesson-learned", "thought-leader"}

    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        if not groq_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        if not tavily_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

        self.llm = LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=groq_key,
            temperature=0.7,
            max_tokens=4096,
        )
        os.environ["GROQ_API_KEY"] = groq_key

        self.tavily_client = TavilyClient(api_key=tavily_key)

        # Create agents
        self.research_agent = self.create_research_agent()
        self.writer_agent = self.create_writer_agent()
        self.validator_agent = self.create_validator_agent()

    def create_research_agent(self):
        return Agent(
            role="LinkedIn Research Specialist",
            goal="Research and gather relevant information about topics, trends, and industry insights to create compelling data-driven answers",
            backstory="Expert researcher with deep knowledge of social media trends and professional content creation. Finds credible statistics, trend analysis, and numerous creative angles to make posts timely and authoritative with conviction.",
            verbose=True,
            llm=self.llm,
        )

    def create_writer_agent(self):
        return Agent(
            role="LinkedIn Content Writer",
            goal="Create engaging LinkedIn posts that will drive engagement and grow user following",
            backstory="Professional copywriter specializing in LinkedIn content with deep understanding of content creation and LinkedIn best practices for posts.",
            verbose=True,
            llm=self.llm,
        )

    def create_validator_agent(self):
        return Agent(
            role="Content Quality Validator",
            goal="Ensure posts are accurate, authentic, and optimized for LinkedIn engagement",
            backstory="Quality assurance expert who catches inaccuracies and ensures factually correct, professionally authentic responses for posts.",
            verbose=True,
            llm=self.llm,
        )

    # --- Helpers ---

    def _validate_input(self, text, max_length=500):
        """Sanitize user input to prevent prompt injection."""
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        text = text.strip()
        if len(text) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length} characters")
        return text

    def _parse_json_response(self, response_str):
        """Extract and parse JSON from LLM output, stripping markdown code fences."""
        result_str = str(response_str)
        if "```json" in result_str:
            result_str = result_str.split("```json")[1].split("```")[0].strip()
        elif "```" in result_str:
            result_str = result_str.split("```")[1].split("```")[0].strip()
        else:
            # fallback: extract first {...} block in case LLM adds extra text
            match = re.search(r"\{.*\}", result_str, re.DOTALL)
            if match:
                result_str = match.group(0)
        return json.loads(result_str, strict=False)

    def _is_rate_limit_error(self, e):
        err = str(e).lower()
        return any(k in err for k in ("429", "resource_exhausted", "rate", "quota"))

    def _is_daily_quota_exhausted(self, e):
        return "GenerateRequestsPerDayPerProjectPerModel" in str(e)

    def _parse_retry_delay(self, e, fallback=60):
        match = re.search(r'retryDelay.*?(\d+)s', str(e))
        return int(match.group(1)) + 5 if match else fallback

    def _run_with_retry(self, crew, max_retries=5, initial_wait=15):
        """Run a crew with exponential backoff on rate-limit errors."""
        for attempt in range(max_retries):
            try:
                return crew.kickoff()
            except Exception as e:
                if not self._is_rate_limit_error(e):
                    raise
                if self._is_daily_quota_exhausted(e):
                    print("Daily quota exhausted. Please try again tomorrow or add billing to your Google AI account.")
                    raise
                if attempt < max_retries - 1:
                    wait_time = self._parse_retry_delay(e, fallback=initial_wait * (2 ** attempt))
                    print(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                else:
                    print("Max retries exceeded. Please try again later.")
                    raise

    # --- Research ---

    def research_topic(self, topic: str, post_type: str) -> Dict:
        """Research topic using Tavily API."""
        try:
            topic = self._validate_input(topic)
            queries = [
                f"{topic} {post_type} latest trends and insights",
                f"{topic} industry statistics recent",
                f"What's new in {topic} this year",
                f"{topic} future outlook and predictions",
                f"key challenges and opportunities in {topic}",
            ]

            research_data = {
                "topic": topic,
                "key_facts": [],
                "trending_angles": [],
                "sources": [],
            }

            for query in queries:
                response = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=2,
                )
                for result in response.get("results", []):
                    research_data["sources"].append({
                        "title": result.get("title"),
                        "url": result.get("url"),
                        "content": result.get("content", "")[:200] + "...",
                    })

            return research_data

        except Exception as e:
            print(f"Research failed: {e}")
            return {"topic": topic, "key_facts": [], "trending_angles": [], "sources": []}

    # --- Tasks ---

    def create_research_task(self, topic: str, post_type: str, research_data: Dict):
        return Task(
            description=f"""
Analyze research data and extract key insights for a LinkedIn post.

Topic: {topic}
Post Type: {post_type}
Research Data: {json.dumps(research_data, indent=2)}

Extract and return JSON:
{{
    "key_facts": ["3-5 compelling statistics and facts"],
    "trending_angles": ["2-3 current angles or perspectives"],
    "credible_sources": ["sources that can be referenced"],
    "hook_opportunities": ["potential opening lines based on research"]
}}

Focus on recent, credible information that supports engaging content creation.
            """,
            agent=self.research_agent,
            expected_output="JSON response with research insights",
        )

    def create_writing_task(self, topic: str, tone: str, post_type: str, research_insights: str):
        return Task(
            description=f"""Create an engaging LinkedIn post based on research insights.

Topic: {topic}
Tone: {tone}
Post Type: {post_type}
Research Insights: {research_insights}

LinkedIn Best Practices:
1. First 1-2 lines must contain a hook that creates curiosity or tension
2. Use short paragraphs (1-2 lines each) to avoid text walls
3. Keep total length between 100-200 words for highest completion rate
4. Use line breaks every 1-2 sentences to create rhythm and readability
5. Avoid using emojis in the post
6. End with a specific question, not a generic thought
7. Tag only when relevant (1-2 people max)
8. Use 4-5 hashtags, but only at the bottom
9. No hashtags in the middle or towards the first line
10. Post must be fair, objective, and respectful of all cultures — no stereotypes,
    discriminatory language, abuses, or slurs

Return JSON:
{{
    "post_content": "Complete LinkedIn post content",
    "hook": "first line extracted",
    "word_count": number,
    "facts_used": ["research points incorporated"]
}}
            """,
            agent=self.writer_agent,
            expected_output="JSON with LinkedIn post content",
        )

    def create_validation_task(self, post_content: str, research_insights: str, tone: str):
        return Task(
            description=f"""
Validate the LinkedIn post for accuracy, quality, and creativity.

Post Content: {post_content}
Original Research: {research_insights}
Tone: {tone}

Validation Criteria:
1. Fact-check all statistics and claims against provided sources
2. Ensure content is original and not copied from sources
3. Verify post follows LinkedIn best practices
4. Check for proper formatting and readability
5. Confirm engagement elements are appropriate

Cringe Flags to Avoid:
- "I'm humbled to announce"
- "Excited to share"
- Generic motivational quotes

Return JSON:
{{
    "validation_passed": true/false,
    "score": "1-10 rating",
    "accuracy_issues": ["any factual issues"],
    "quality_issues": ["areas of improvement"],
    "cringe_flags": ["cliche phrases found"],
    "suggestions": ["specific improvements"],
    "final_verdict": "ready to post / needs revision"
}}
            """,
            agent=self.validator_agent,
            expected_output="Validation report with feedback in JSON",
        )
        
    def generate_post(self,topic,tone,post_type):
        topic = self._validate_input(topic)
        
        if tone not in self.VALID_TONES:
            tone = "professional"
        if post_type not in self.VALID_TYPES:
            post_type = "thought-leader"
            
        print(f"Generating LinkedIn post about :{topic}")
        print(f"Tone:{tone} | Type :{post_type}")
        print('='*50)
        
        #Step 1 : Research 
        print(f"Step-1 : Researching the topic........\n")
        research_data = self.research_topic(topic, post_type)
        research_task = self.create_research_task(topic, post_type, research_data)
        research_crew = Crew(
            agents=[self.research_agent],
            tasks=[research_task],
            process=Process.sequential,
            verbose=False,
        )
        research_result = self._run_with_retry(research_crew)
        print(f"Research completed.\n")
        
        #Step 2 : Write
        print(f"Step-2 : Writing the post........\n")
        writing_task = self.create_writing_task(topic, tone, post_type, str(research_result))
        writing_crew = Crew(
            agents=[self.writer_agent],
            tasks=[writing_task],
            process=Process.sequential,
            verbose=False,
        )
        writing_result = self._run_with_retry(writing_crew)
        print(f"Writing completed.\n")
        
        try:
            post_data = self._parse_json_response(writing_result)
        except Exception as e:
            print(f"Error parsing writing result: {e}")
            return None
        
        #Step 3 : Validate
        print(f"Step 3 : Validating posts......\n")
        validation_task = self.create_validation_task(post_data["post_content"], str(research_result), tone)
        validation_crew = Crew(
            agents=[self.validator_agent],
            tasks=[validation_task],
            process=Process.sequential,
            verbose=False,
        )
        validation_result = self._run_with_retry(validation_crew)
        print(f"Validation Completed")
        
        try:
            validation_data = self._parse_json_response(validation_result)
        except Exception as e:
            print(f"Error parsing validation result: {e}")
            validation_data = {"validation_passed":False, "score":0}
            
        return {
            "post":post_data,
            "validation":validation_data,
            "research_sources":len(research_data.get('sources',[])),
        }
        
        
def main():
    generator = LinkedInPostGenerator()
    
    print("Linkedin Post Generator")
    print(f"="*50)
    
    topic = input("Enter your topic/idea: ").strip()
    if not topic:
        print(f"Topic is required")
        return 
    
    print(f"\nTone options {generator.VALID_TONES}")
    tone = input("Choose tone (default: professional): ").strip().lower() or "professional"
    
    print(f"\nPost type options: {generator.VALID_TYPES}")
    post_type = input("Choose post type (default: thought-leader): ").strip().lower() or "thought-leader"
    
    result = generator.generate_post(topic, tone, post_type)
    
    if result:
        print(f"\n{'='*50}")
        print(f"GENERATED LINKEDIN POST.")
        print(f"{'='*50}")
        print(result['post']['post_content'])
        print(f"\n{'='*50}")
        print(f"Word Count : {result['post']['word_count']}")
        print(f"Validation Score : {result['validation']['score']}/10")
        print(f"Research Sources used : {result['research_sources']}")
        
        if result['validation'].get('suggestions'):
            print(f"Suggestions :")
            for suggestion in result['validation']['suggestions']:
                print(f"  - {suggestion}")
                
        print(f"\nStatus : {result['validation'].get('final_verdict', 'unknown')}")
    else:
        print("Failed to generate post.")
        
if __name__ == "__main__":
    main()