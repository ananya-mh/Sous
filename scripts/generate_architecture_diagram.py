from graphviz import Digraph

def create_ieee_architecture_diagram():
    """Create a high-quality technical system architecture for IEEE publication"""
    # Use 'dot' engine for hierarchical layout
    dot = Digraph(comment='Sous System Architecture IEEE', format='png', engine='dot')
    
    # Global Attributes: Font is crucial for IEEE (Arial or Helvetica)
    dot.attr(rankdir='TB', size='7.5,10', ratio='auto', margin='0.2', dpi='600')
    dot.attr('node', shape='rectangle', style='filled', fontname='Helvetica,Arial', fontsize='10', height='0.5', width='1.5')
    dot.attr('edge', fontname='Helvetica,Arial', fontsize='9', arrowsize='0.7')
    
    # 1. External Actor
    dot.node('user', 'Client Interface\n(Web Browser)', shape='box3d', fillcolor='#f0f0f0')

    # 2. Presentation Layer
    with dot.subgraph(name='cluster_0') as c:  # pyright: ignore[reportOptionalContextManager]
        c.attr(label='Presentation Layer', fontname='Helvetica-Bold', fontsize='11', style='dotted')
        c.node('react', 'React v19 SPA\n(UI/UX Components)', fillcolor='#e3f2fd')
        c.node('state', 'State Management\n(Hooks/LocalStorage)', fillcolor='#e3f2fd')

    # 3. Application Layer
    with dot.subgraph(name='cluster_1') as c:  # pyright: ignore[reportOptionalContextManager]
        c.attr(label='Service Layer (FastAPI)', fontname='Helvetica-Bold', fontsize='11', style='dotted')
        c.node('api_gateway', 'REST API Endpoints\n(Auth/Parse/Sub)', fillcolor='#f1f8e9')
        c.node('security', 'JWT Manager\n(bcrypt/HS256)', fillcolor='#f1f8e9')

    # 4. Intelligence & Model Layer
    with dot.subgraph(name='cluster_2') as c:  # pyright: ignore[reportOptionalContextManager]
        c.attr(label='AI Processing Pipeline', fontname='Helvetica-Bold', fontsize='11', style='dotted')
        c.node('bert_engine', 'Fine-tuned BERT\n(NER Inference)', fillcolor='#fff3e0')
        c.node('llm_api', 'LLM via Groq (gpt-oss-120b)\n(Substitution Logic)', fillcolor='#fff3e0')
        c.node('nlp_proc', 'NLP Pre-processor\n(Token Alignment)', fillcolor='#fff3e0')

    # 5. Data Persistence Layer
    with dot.subgraph(name='cluster_3') as c:  # pyright: ignore[reportOptionalContextManager]
        c.attr(label='Persistence Layer', fontname='Helvetica-Bold', fontsize='11', style='dotted')
        c.node('db', 'MySQL RDBMS\n(User/Schema)', shape='cylinder', fillcolor='#f3e5f5')
        c.node('weights', 'Local Model Weights\n(PyTorch Binary)', shape='folder', fillcolor='#f3e5f5')

    # Logical Connections with technical annotations
    dot.edge('user', 'react', label=' HTTPS/TLS')
    dot.edge('react', 'state', style='dashed', dir='both')
    
    # Interface between Frontend and Backend
    dot.edge('react', 'api_gateway', label=' REST API (JSON)')
    dot.edge('api_gateway', 'security', dir='both')
    
    # Internal Backend Logic
    dot.edge('security', 'db', label=' SQL/PyMySQL')
    dot.edge('api_gateway', 'nlp_proc', label=' Raw Text')
    dot.edge('nlp_proc', 'bert_engine')
    dot.edge('nlp_proc', 'llm_api')
    
    # Data dependencies
    dot.edge('weights', 'bert_engine', style='dashed')
    
    # Return flows (Simplified for clarity)
    dot.edge('bert_engine', 'api_gateway', label=' Entities', color='#666666')
    dot.edge('llm_api', 'api_gateway', label=' JSON Sug.', color='#666666')

    return dot

if __name__ == '__main__':
    dot = create_ieee_architecture_diagram()
    dot.render('Sous_Architecture_IEEE', cleanup=True)
    print("IEEE Professional Diagram Generated: Sous_Architecture_IEEE.png")