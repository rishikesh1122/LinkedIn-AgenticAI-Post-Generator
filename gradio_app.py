import gradio as gr
from linkedin_generator import LinkedInPostGenerator
import os

def generate_linkedin_post(topic, tone, post_type):
    if not topic:
        return "", "", "", "Please Enter a topic <Mandatory>"
    
    try:
        generator = LinkedInPostGenerator()
        result = generator.generate_post(topic, tone, post_type)
        
        if result:
            post_content = result['post']['post_content']
            word_count = result['post'].get('word_count','N/A')
            score = result['validation'].get('score','N/A')
            suggestions = result['validation'].get('suggestions',[])
            
            stats = f"Word Count:{word_count} | Validation Score : {score}/10"
            suggestions_text = "\n".join([f" ->  {s}" for s in suggestions]) if suggestions else "No Suggestions"
            
            return post_content, stats, suggestions_text, "Post generated successfully!"
        else:
            return "Failed to generate post.", "", "", "Generation failed."
    except Exception as e:
        return f"Error: {str(e)}", "", "", "An error occurred."
    
    

#gradio frontend
with gr.Blocks(title="LinkedIn Post Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# LinkedIn Post Generator")
    gr.Markdown("Generate professional LinkedIn post using AI-agents research, writing and validation agents")
    
    with gr.Row():
        with gr.Column(scale=1):
            topic_input = gr.Textbox(
                label='Topic',
                placeholder="eg., AI in healthcare, Remote work trends....",
                lines=2
            )
            
            tone_input = gr.Radio(
                choices=['professional','casual','thought-leader'],
                value='thought-leader',
                label='Tone'
            )
            
            post_type_input = gr.Radio(
                choices=["story", "hot-take", "announcement", "lesson-learned", "thought-leader"],
                value='thought-leader',
                label='Post Type'
            )
            
            generate_btn = gr.Button("Generate Post", variant='primary',size='lg')
    
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Status", interactive=False)
            post_output = gr.Textbox(label="Generated LinkedIn Post", interactive=False, lines=15)
            stats_output = gr.Textbox(label="Stats", interactive=False)
            suggestions_output = gr.Textbox(label="Suggestions", interactive=False,lines=5)
            
    generate_btn.click(
        fn=generate_linkedin_post,
        inputs=[topic_input, tone_input, post_type_input],
        outputs=[post_output, stats_output, suggestions_output, status_output]
    )
    
    gr.Markdown("-------")
    gr.Markdown("TIP : The genrator uses research agents to find recent data and trends about your topic: ")
    
    
# #dunder methods
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )