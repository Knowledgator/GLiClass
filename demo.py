"""
GLiClass Enhanced Demo with Advanced Features

Features:
- Task description prompts
- Hierarchical label inputs (JSON format)
- Few-shot examples
- Hierarchical output structure
- Label descriptions
"""

import json
from typing import Dict, List, Any, Union, Optional
import gradio as gr
import torch
from transformers import AutoTokenizer

from gliclass import GLiClassModel, ZeroShotClassificationPipeline

# Initialize model and pipeline
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_path = "knowledgator/gliclass-small-v1.0"
model = GLiClassModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, 
    classification_type='multi-label', 
    device=device
)

# ============== Example Texts ==============

TEXT_PRODUCT_REVIEW = """
I recently purchased the Sony WH-1000XM4 Wireless Noise-Canceling Headphones from Amazon and I must say, I'm thoroughly impressed. The package arrived in New York within 2 days, thanks to Amazon Prime's expedited shipping.

The headphones themselves are remarkable. The noise-canceling feature works like a charm in the bustling city environment, and the 30-hour battery life means I don't have to charge them every day. Connecting them to my Samsung Galaxy S21 was a breeze, and the sound quality is second to none.

I also appreciated the customer service from Amazon when I had a question about the warranty. They responded within an hour and provided all the information I needed.

However, the headphones did not come with a hard case, which was listed in the product description. I contacted Amazon, and they offered a 10% discount on my next purchase as an apology.

Overall, I'd give these headphones a 4.5/5 rating and highly recommend them to anyone looking for top-notch quality in both product and service.
"""

TEXT_TECH_COMPANIES = """
Apple Inc. is an American multinational technology company headquartered in Cupertino, California. Apple is the world's largest technology company by revenue, with US$394.3 billion in 2022 revenue. As of March 2023, Apple is the world's biggest company by market capitalization.

Microsoft was founded by Bill Gates and Paul Allen on April 4, 1975 to develop and sell BASIC interpreters for the Altair 8800. During his career at Microsoft, Gates held the positions of chairman, chief executive officer, president and chief software architect.

Apple was founded as Apple Computer Company on April 1, 1976, by Steve Wozniak, Steve Jobs (1955–2011) and Ronald Wayne to develop and sell Wozniak's Apple I personal computer.
"""

TEXT_SCIENTIFIC = """
Several studies have reported its pharmacological activities, including anti-inflammatory, antimicrobial, and antitumoral effects. 
The effect of E-anethole was studied in the osteosarcoma MG-63 cell line, and the antiproliferative activity was evaluated by an MTT assay. 
It showed a GI50 value of 60.25 μM with apoptosis induction through the mitochondrial-mediated pathway. Additionally, it induced cell cycle arrest at the G0/G1 phase, up-regulated the expression of p53, caspase-3, and caspase-9, and down-regulated Bcl-xL expression.
"""

TEXT_RESTAURANT_REVIEW = """
We visited La Maison last Friday for our anniversary dinner. The ambiance was absolutely stunning - dim lighting, soft jazz music, and elegant table settings. Our waiter, Marcus, was incredibly attentive without being intrusive.

For appetizers, we had the truffle bruschetta and the soup of the day. Both were divine! The main courses - filet mignon for me and lobster risotto for my wife - were cooked to perfection. 

The only downside was the wait time for our desserts, which took about 25 minutes. However, the chocolate soufflé was worth the wait!

Price was on the higher side ($180 for two), but the quality justified the cost. Will definitely return!
"""

TEXT_NEWS_POLITICS = """
The Senate passed a landmark bipartisan infrastructure bill late Thursday night, allocating $1.2 trillion for roads, bridges, broadband internet, and clean energy initiatives. The vote was 69-30, with 19 Republican senators joining all Democrats in support.

President Biden called the passage "a historic investment in America's future" and urged the House to act quickly. However, progressive Democrats have signaled they won't vote for the infrastructure bill unless it's paired with a larger social spending package.

Senate Minority Leader criticized portions of the bill related to climate spending, calling them "unnecessary green new deal provisions," while environmental groups praised the clean energy investments as "a step in the right direction, but not nearly enough."
"""

TEXT_SPORTS = """
In a thrilling overtime finish, the Lakers defeated the Celtics 118-112 in Game 7 of the NBA Finals. LeBron James delivered a historic performance with 42 points, 16 rebounds, and 10 assists, securing his fifth championship ring and fourth Finals MVP award.

The game was tied at 102 with 30 seconds remaining in regulation when Marcus Smart hit a contested three-pointer. However, James answered with a driving layup at the buzzer to force overtime.

In the extra period, the Lakers outscored Boston 16-10, with Anthony Davis contributing two crucial blocks in the final minute. "This is what you dream about as a kid," James said in the post-game interview. "Playing against the Celtics, Game 7, everything on the line."
"""

TEXT_MOVIE_REVIEW = """
Christopher Nolan's "Oppenheimer" is a masterwork of biographical cinema that demands to be seen on the largest screen possible. Cillian Murphy delivers a career-defining performance as J. Robert Oppenheimer, capturing both the brilliance and moral anguish of the father of the atomic bomb.

The film's nonlinear structure, weaving between the Manhattan Project, the 1954 security hearing, and the 1959 Lewis Strauss confirmation hearing, could have been confusing. Instead, Nolan crafts a compelling narrative that builds to a devastating emotional climax.

At three hours, some viewers may find the pacing challenging, particularly in the courtroom sequences. However, the technical achievements - Ludwig Göransson's haunting score, Hoyte van Hoytema's IMAX cinematography - make this an unmissable theatrical experience. Rating: 9/10
"""

TEXT_TECH_STARTUP = """
San Francisco-based AI startup Anthropic announced today it has raised $450 million in Series C funding, valuing the company at $5 billion. The round was led by Spark Capital, with participation from Google and existing investors.

Founded in 2021 by former OpenAI researchers Dario and Daniela Amodei, Anthropic has positioned itself as a leader in AI safety research. The company's Claude assistant has gained significant market share in the enterprise segment.

"This funding will accelerate our research into interpretable and steerable AI systems," said CEO Dario Amodei. "We believe safety and capability go hand in hand." The company plans to double its research team and expand internationally, with offices planned in London and Tokyo.
"""

TEXT_HEALTH_WELLNESS = """
A new study published in the Journal of the American Medical Association suggests that intermittent fasting may offer significant benefits beyond weight loss. Researchers followed 500 participants over two years and found improvements in cardiovascular health markers, insulin sensitivity, and cognitive function.

Participants who followed a 16:8 fasting protocol (eating within an 8-hour window) showed a 15% reduction in LDL cholesterol and a 20% improvement in fasting glucose levels compared to the control group.

However, experts caution that intermittent fasting isn't suitable for everyone. "Pregnant women, people with a history of eating disorders, and those with certain medical conditions should consult their doctor first," said Dr. Sarah Chen, the study's lead author. "It's not a magic solution, but for many people, it can be a sustainable approach to improving metabolic health."
"""

TEXT_TRAVEL = """
Hidden among the limestone karsts of Ha Long Bay, Cat Ba Island offers travelers an authentic Vietnamese experience away from the tourist crowds. We spent five days exploring this gem and discovered why it's becoming a favorite among backpackers and adventure seekers.

The island's national park features challenging hikes through tropical rainforest, with the trek to the peak of Ngu Lam offering panoramic views of the bay. We also kayaked through hidden lagoons and explored caves that few tourists ever see.

Accommodation ranges from basic hostels ($8/night) to comfortable eco-resorts ($60/night). The seafood is incredibly fresh - we had the best grilled squid of our lives at a family-run restaurant in Cat Ba Town for just $5. Pro tip: rent a motorbike to explore the quieter beaches on the island's east side.
"""

TEXT_COOKING_RECIPE = """
This Thai green curry comes together in just 30 minutes and tastes better than takeout. The secret is making your own curry paste - it takes an extra 10 minutes but the flavor difference is remarkable.

For the paste, blend together: 10 green chilies, 4 garlic cloves, 2 shallots, 1 stalk lemongrass, 1 inch galangal, handful of cilantro stems, 1 tsp cumin, 1 tsp coriander, zest of 1 lime, and 2 tbsp fish sauce. 

Heat coconut oil in a wok, fry the paste for 2 minutes until fragrant. Add chicken (or tofu), cook until browned. Pour in coconut milk, add bamboo shoots, Thai eggplant, and basil. Simmer for 15 minutes. Season with palm sugar and more fish sauce to taste.

Serve over jasmine rice with extra chilies on the side. This recipe serves 4 and can be made ahead - the flavors actually improve overnight.
"""

TEXT_FINANCIAL_ADVICE = """
With inflation running at 4.2% and the Fed signaling more rate hikes, many investors are wondering how to position their portfolios. Here's what our analysis suggests for Q4 2024.

Fixed income is finally attractive again. With 10-year Treasury yields above 4.5%, bonds offer meaningful real returns for the first time in years. We recommend increasing allocation to investment-grade corporate bonds and TIPS for inflation protection.

For equities, we're cautiously optimistic on value stocks, particularly in the energy and financial sectors. Tech valuations remain stretched despite recent pullbacks. International developed markets, especially Japan and Europe, offer better risk-reward at current levels.

Remember: past performance doesn't guarantee future results. This is general information, not personalized advice. Consult a financial advisor before making investment decisions.
"""

TEXT_ENVIRONMENTAL = """
The Great Barrier Reef experienced its sixth mass bleaching event in a decade this summer, with aerial surveys showing 91% of reefs affected. Scientists warn that without dramatic action on climate change, the world's largest coral ecosystem may not survive beyond 2050.

"We're witnessing the collapse of one of Earth's most biodiverse ecosystems in real time," said Dr. Terry Hughes of James Cook University. Water temperatures reached 2°C above the February average, causing corals to expel the symbiotic algae that give them color and nutrients.

Some researchers are experimenting with heat-resistant coral varieties and cloud-brightening technology to shade reefs. However, most scientists agree these are stopgap measures. "The only real solution is rapid decarbonization," Hughes said. "Everything else is just buying time."
"""

TEXT_EDUCATION = """
The debate over standardized testing in American schools has intensified following a new report showing significant post-pandemic learning gaps. The National Assessment of Educational Progress found that fourth-grade math scores dropped to levels not seen since 2005.

Proponents of testing argue that standardized assessments are essential for identifying struggling students and holding schools accountable. "Without data, we're flying blind," said Education Secretary Miguel Cardona. "Tests help us direct resources where they're needed most."

Critics counter that high-stakes testing narrows the curriculum and increases student stress without improving outcomes. "We're testing kids more than ever, but educational outcomes aren't improving," said education researcher Dr. Pasi Sahlberg. "Countries like Finland, which use minimal standardized testing, consistently outperform the US."
"""

TEXT_FASHION = """
Milan Fashion Week wrapped up yesterday with several surprising trends that will likely dominate fall/winter 2025. After years of quiet luxury and minimalism, designers are embracing bold maximalism - think dramatic volumes, clashing prints, and unapologetic color.

Prada's collection featured oversized coats with exaggerated shoulders paired with flowing silk pants, while Gucci returned to its pattern-mixing roots under new creative direction. Versace went full baroque with gold-embroidered gowns that would feel at home in a Renaissance painting.

Sustainability remained a talking point, with Stella McCartney showcasing a collection made entirely from recycled ocean plastic. However, critics noted that the industry still has far to go. "One sustainable collection doesn't offset the environmental impact of fast fashion," noted fashion journalist Vanessa Friedman. "The industry needs systemic change, not just good PR."
"""

TEXT_LEGAL_CASE = """
The Supreme Court agreed Monday to hear a case that could reshape the boundaries of free speech on social media platforms. The case, NetChoice v. Paxton, challenges Texas and Florida laws that prohibit large social media companies from removing certain political content.

Tech companies argue that the First Amendment protects their right to moderate content as they see fit, similar to how newspapers decide what to publish. "Forcing platforms to host speech they find objectionable is compelled speech, which the Constitution forbids," said NetChoice counsel Paul Clement.

Texas and Florida counter that social media platforms function as common carriers or public utilities and should be subject to similar non-discrimination requirements. "These companies have become the modern public square," said Texas Attorney General Ken Paxton. "They shouldn't be able to silence voices based on political viewpoint."
"""

TEXT_GAMING = """
After three years in development hell, "Hollow Eclipse" has finally launched - and it's everything fans hoped for. This action RPG from indie studio Moonlight Games delivers a haunting 40-hour adventure that rivals titles from studios with ten times the budget.

The combat system strikes a perfect balance between accessibility and depth. Basic attacks and dodges are simple to execute, but mastering the "shadow merge" mechanic - which lets you temporarily possess enemies - adds layers of strategy. Boss fights are challenging without feeling unfair, though the final boss may take even experienced players dozens of attempts.

Where the game truly shines is its atmosphere. The decaying gothic city of Velmoor is rendered in stunning hand-drawn art, and the ambient soundtrack creates constant unease. The story tackles themes of grief and memory with surprising emotional maturity. Minor technical issues (occasional frame drops, one softlock) can't diminish this achievement. Score: 9.5/10
"""

TEXT_REAL_ESTATE = """
The housing market is sending mixed signals as we enter 2025. Existing home sales fell for the third consecutive month, down 4.1% in November, yet prices continue to climb in most metropolitan areas. The median home price hit $416,000, up 3.8% year-over-year.

Low inventory remains the central issue. Many homeowners are reluctant to sell because they've locked in sub-3% mortgage rates and don't want to trade up to today's 7% rates. This "lock-in effect" has created a severe shortage of listings, particularly in the starter home category.

"We're seeing bidding wars even in this high-rate environment because there's simply nothing to buy," said economist Lawrence Yun. First-time buyers are particularly squeezed, with affordability at its worst level since 1984. Some markets, including Austin and Phoenix, are showing price corrections, but coastal cities remain stubbornly expensive.
"""

TEXT_MENTAL_HEALTH = """
Workplace burnout has reached epidemic proportions, with a new Gallup survey finding that 76% of employees experience burnout at least sometimes. But recognizing burnout isn't always straightforward - it often manifests differently than simple exhaustion.

The three hallmarks of burnout are: emotional exhaustion (feeling drained and unable to cope), depersonalization (becoming cynical and detached from work), and reduced personal accomplishment (feeling ineffective regardless of actual performance).

Recovery requires more than a vacation. "You can't just rest your way out of burnout," says psychologist Dr. Christina Maslach, who pioneered burnout research. "You need to address the root causes - usually workload, lack of control, insufficient recognition, or values conflicts." Strategies include setting firm boundaries, delegating tasks, and having honest conversations with managers about sustainable workloads. In severe cases, professional support from a therapist can help.
"""

TEXT_ASTRONOMY = """
NASA's James Webb Space Telescope has detected what may be signs of biological activity in the atmosphere of K2-18b, an exoplanet 120 light-years away. The discovery has electrified the scientific community, though researchers caution against jumping to conclusions.

The telescope's spectrometers identified dimethyl sulfide (DMS), a molecule produced almost exclusively by living organisms on Earth. Webb also confirmed the presence of methane and carbon dioxide, consistent with a water-rich atmosphere.

"This is tantalizing, but not definitive proof of life," said lead researcher Dr. Nikku Madhusudhan. "DMS could potentially be produced by unknown geological processes. We need more observations." K2-18b is a "Hycean" world - a planet with a hydrogen-rich atmosphere and potentially a liquid water ocean beneath. If confirmed, this would be humanity's first detection of a potential biosignature beyond our solar system.
"""


def parse_labels_input(labels_input: str) -> Union[List[str], Dict[str, Any]]:
    """
    Parse labels input - supports both comma-separated and JSON hierarchical format.
    
    Examples:
    - "positive, negative, neutral" -> ["positive", "negative", "neutral"]
    - '{"sentiment": ["positive", "negative"], "topic": ["food", "service"]}' -> dict
    """
    labels_input = labels_input.strip()
    
    # Try parsing as JSON first (for hierarchical labels)
    if labels_input.startswith('{'):
        try:
            return json.loads(labels_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for hierarchical labels: {e}")
    
    # Otherwise, treat as comma-separated flat labels
    labels = [label.strip() for label in labels_input.split(',') if label.strip()]
    return labels


def parse_examples_input(examples_input: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse few-shot examples input (JSON format).
    
    Expected format:
    [
        {"text": "Example text 1", "labels": ["label1", "label2"]},
        {"text": "Example text 2", "labels": ["label3"]}
    ]
    """
    if not examples_input or not examples_input.strip():
        return None
    
    try:
        examples = json.loads(examples_input.strip())
        if not isinstance(examples, list):
            raise ValueError("Examples must be a JSON array")
        
        for i, ex in enumerate(examples):
            if not isinstance(ex, dict):
                raise ValueError(f"Example {i+1} must be a JSON object")
            if 'text' not in ex:
                raise ValueError(f"Example {i+1} missing 'text' field")
            if 'labels' not in ex and 'true_labels' not in ex:
                raise ValueError(f"Example {i+1} missing 'labels' field")
        
        return examples
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format for examples: {e}")


def format_output(
    results: Union[List[Dict], Dict], 
    hierarchical: bool = False,
    output_format: str = "visual"
) -> Union[Dict[str, float], str]:
    """Format classification output for Gradio display."""
    
    if output_format == "json":
        return format_as_json(results, hierarchical)
    
    if hierarchical and isinstance(results, dict):
        # Format hierarchical output as readable string
        return format_hierarchical_dict(results)
    
    if isinstance(results, list):
        return {result['label']: float(result['score']) for result in results}
    
    return results


def format_as_json(results: Union[List[Dict], Dict], hierarchical: bool = False) -> str:
    """Format results as pretty-printed JSON string."""
    if hierarchical and isinstance(results, dict):
        # Already in hierarchical dict format
        return json.dumps(results, indent=2, ensure_ascii=False)
    
    if isinstance(results, list):
        # Convert list of predictions to structured format
        output = {
            "predictions": [
                {"label": r["label"], "score": round(r["score"], 4)}
                for r in results
            ],
            "scores": {r["label"]: round(r["score"], 4) for r in results}
        }
        return json.dumps(output, indent=2, ensure_ascii=False)
    
    return json.dumps(results, indent=2, ensure_ascii=False)


def format_hierarchical_dict(d: Dict, indent: int = 0) -> str:
    """Format hierarchical dict for display with visual score bars."""
    lines = []
    prefix = "  " * indent
    
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}**{key}**:")
            lines.append(format_hierarchical_dict(value, indent + 1))
        else:
            score_bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
            lines.append(f"{prefix}{key}: {score_bar} {value:.3f}")
    
    return "\n".join(lines)


def classification(
    text: str,
    labels_input: str,
    threshold: float,
    multi_label: bool,
    prompt: str,
    examples_input: str,
    hierarchical_output: bool,
    output_format: str = "visual"
) -> Union[Dict[str, float], str]:
    """
    Perform classification with all advanced features.
    """
    try:
        # Parse labels (flat or hierarchical)
        labels = parse_labels_input(labels_input)
        
        # Parse few-shot examples
        examples = parse_examples_input(examples_input) if examples_input else None
        
        # Set classification type
        pipeline.pipe.classification_type = 'multi-label' if multi_label else 'single-label'
        
        # Prepare prompt
        task_prompt = prompt.strip() if prompt and prompt.strip() else None
        
        # Run classification
        results = pipeline(
            text, 
            labels, 
            threshold=threshold,
            examples=examples,
            prompt=task_prompt,
            return_hierarchical=hierarchical_output
        )[0]  # Single text, get first result
        
        # Format output based on selected format
        if output_format == "json":
            return format_as_json(results, hierarchical_output)
        elif hierarchical_output:
            return format_hierarchical_dict(results)
        else:
            return {result['label']: float(result['score']) for result in results}
            
    except Exception as e:
        return f"Error: {str(e)}"


# ============== Example Configurations ==============

EXAMPLES = [
    # Example 1: Basic flat labels with prompt
    [
        TEXT_PRODUCT_REVIEW,
        "product review, electronics, positive feedback, negative feedback, customer service, shipping",
        0.5,
        True,
        "Classify this customer review by topic and sentiment:",
        "",
        False,
        "visual"
    ],
    # Example 2: Hierarchical labels for restaurant review
    [
        TEXT_RESTAURANT_REVIEW,
        '''{
    "sentiment": ["positive", "negative", "mixed"],
    "aspects": ["food quality", "service", "ambiance", "price", "wait time"],
    "recommendation": ["would recommend", "would not recommend"]
}''',
        0.4,
        True,
        "Analyze this restaurant review:",
        "",
        True,
        "visual"
    ],
    # Example 3: News article with few-shot examples
    [
        TEXT_NEWS_POLITICS,
        "politics, business, technology, sports, entertainment, science, health",
        0.5,
        True,
        "Classify this news article by category:",
        '''[
    {"text": "The Federal Reserve raised interest rates by 0.25% today, citing persistent inflation concerns.", "labels": ["politics", "business"]},
    {"text": "Scientists discover high new high-temperature superconductor material that works at room temperature.", "labels": ["science", "technology"]}
]''',
        False,
        "visual"
    ],
    # Example 4: Scientific classification with hierarchical output
    [
        TEXT_SCIENTIFIC,
        '''{
    "domain": ["biology", "chemistry", "medicine", "physics"],
    "research_type": ["experimental", "theoretical", "review"],
    "application": ["therapeutic", "diagnostic", "basic research"]
}''',
        0.3,
        True,
        "Classify this scientific abstract:",
        "",
        True,
        "visual"
    ],
    # Example 5: Sports article - single label
    [
        TEXT_SPORTS,
        "basketball, football, soccer, tennis, baseball, hockey, golf",
        0.5,
        False,
        "What sport is this article about?",
        "",
        False,
        "visual"
    ],
    # Example 6: Movie review with detailed sentiment (JSON output)
    [
        TEXT_MOVIE_REVIEW,
        '''{
    "overall_sentiment": ["positive", "negative", "mixed"],
    "aspects_praised": ["acting", "direction", "cinematography", "music", "story", "pacing"],
    "aspects_criticized": ["acting", "direction", "cinematography", "music", "story", "pacing"],
    "recommendation": ["must watch", "worth watching", "skip it"]
}''',
        0.35,
        True,
        "Analyze this movie review in detail:",
        "",
        True,
        "json"
    ],
    # Example 7: Tech startup news
    [
        TEXT_TECH_STARTUP,
        "funding announcement, product launch, acquisition, IPO, partnership, hiring, layoffs, legal",
        0.4,
        True,
        "What type of tech news is this?",
        "",
        False,
        "visual"
    ],
    # Example 8: Health article with hierarchical categories
    [
        TEXT_HEALTH_WELLNESS,
        '''{
    "topic": ["nutrition", "exercise", "mental health", "sleep", "medical research"],
    "content_type": ["research findings", "practical advice", "expert opinion", "warning"],
    "audience": ["general public", "healthcare professionals", "patients"]
}''',
        0.4,
        True,
        "Categorize this health article:",
        "",
        True,
        "visual"
    ],
    # Example 9: Travel content (JSON output)
    [
        TEXT_TRAVEL,
        "destination guide, hotel review, restaurant review, adventure travel, budget travel, luxury travel, travel tips",
        0.4,
        True,
        "What type of travel content is this?",
        "",
        False,
        "json"
    ],
    # Example 10: Recipe classification
    [
        TEXT_COOKING_RECIPE,
        '''{
    "cuisine": ["Thai", "Italian", "Mexican", "Indian", "Chinese", "Japanese", "French", "American"],
    "difficulty": ["easy", "medium", "hard"],
    "meal_type": ["breakfast", "lunch", "dinner", "dessert", "snack"],
    "dietary": ["vegetarian friendly", "vegan friendly", "gluten free", "dairy free", "contains meat"]
}''',
        0.35,
        True,
        "Classify this recipe:",
        "",
        True,
        "visual"
    ],
    # Example 11: Financial content with examples
    [
        TEXT_FINANCIAL_ADVICE,
        "investment advice, market analysis, personal finance, retirement planning, tax advice, economic news",
        0.4,
        True,
        "Categorize this financial content:",
        '''[
    {"text": "Here are 5 ways to maximize your 401k contributions before year end.", "labels": ["personal finance", "retirement planning", "tax advice"]},
    {"text": "The S&P 500 rose 2% today following strong jobs report.", "labels": ["market analysis", "economic news"]}
]''',
        False,
        "visual"
    ],
    # Example 12: Environmental news (JSON output)
    [
        TEXT_ENVIRONMENTAL,
        '''{
    "topic": ["climate change", "biodiversity", "pollution", "conservation", "renewable energy"],
    "tone": ["alarming", "hopeful", "neutral", "urgent"],
    "focus": ["problem description", "solutions", "policy", "research findings"]
}''',
        0.35,
        True,
        "Analyze this environmental article:",
        "",
        True,
        "json"
    ],
    # Example 13: Education debate
    [
        TEXT_EDUCATION,
        "education policy, standardized testing, curriculum, teacher issues, student welfare, technology in education, higher education",
        0.4,
        True,
        "What education topics does this article cover?",
        "",
        False,
        "visual"
    ],
    # Example 14: Fashion news with hierarchy
    [
        TEXT_FASHION,
        '''{
    "content_type": ["trend report", "designer profile", "collection review", "industry news", "sustainability"],
    "season": ["spring/summer", "fall/winter"],
    "market_segment": ["luxury", "fast fashion", "sustainable fashion", "streetwear"]
}''',
        0.4,
        True,
        "Classify this fashion article:",
        "",
        True,
        "visual"
    ],
    # Example 15: Legal case (JSON output)
    [
        TEXT_LEGAL_CASE,
        "constitutional law, criminal law, civil rights, corporate law, intellectual property, free speech, privacy",
        0.4,
        True,
        "What areas of law does this case involve?",
        "",
        False,
        "json"
    ],
    # Example 16: Gaming review with detailed analysis
    [
        TEXT_GAMING,
        '''{
    "genre": ["action", "RPG", "adventure", "puzzle", "strategy", "simulation", "sports"],
    "platform_feel": ["indie", "AAA", "mid-tier"],
    "strengths": ["gameplay", "story", "graphics", "music", "replayability"],
    "weaknesses": ["bugs", "difficulty", "length", "graphics", "story"],
    "recommendation": ["must play", "worth playing", "wait for sale", "skip"]
}''',
        0.35,
        True,
        "Analyze this game review:",
        "",
        True,
        "visual"
    ],
    # Example 17: Real estate market analysis
    [
        TEXT_REAL_ESTATE,
        "market analysis, buying advice, selling advice, investment, rental market, mortgage rates, housing policy",
        0.4,
        True,
        "What real estate topics are covered?",
        "",
        False,
        "visual"
    ],
    # Example 18: Mental health with few-shot (JSON output)
    [
        TEXT_MENTAL_HEALTH,
        '''{
    "topic": ["burnout", "anxiety", "depression", "stress management", "work-life balance"],
    "content_type": ["educational", "self-help advice", "research summary", "personal story"],
    "actionability": ["provides concrete steps", "general awareness", "seeks professional help"]
}''',
        0.35,
        True,
        "Categorize this mental health content:",
        '''[
    {"text": "Feeling overwhelmed? Try the 5-4-3-2-1 grounding technique: notice 5 things you see, 4 you hear...", "labels": ["topic.anxiety", "topic.stress management", "content_type.self-help advice", "actionability.provides concrete steps"]},
    {"text": "A new study links social media use exceeding 3 hours daily with increased rates of depression in teens.", "labels": ["topic.depression", "content_type.research summary", "actionability.general awareness"]}
]''',
        True,
        "json"
    ],
    # Example 19: Astronomy discovery
    [
        TEXT_ASTRONOMY,
        "exoplanets, astrobiology, cosmology, solar system, space exploration, telescopes, astrophysics",
        0.4,
        True,
        "What astronomy topics are discussed?",
        "",
        False,
        "visual"
    ],
    # Example 20: Tech companies - single label
    [
        TEXT_TECH_COMPANIES,
        "company profile, product announcement, financial report, industry analysis, biography, opinion piece",
        0.5,
        False,
        "What is the primary type of this article?",
        "",
        False,
        "visual"
    ],
]


# ============== Gradio Interface ==============

with gr.Blocks(
    title="GLiClass Advanced Demo",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )
) as demo:
    
    gr.Markdown("""
    # 🏷️ GLiClass Advanced Zero-Shot Classification
    
    Enhanced demo featuring **prompts**, **hierarchical labels**, **few-shot examples**, and **structured outputs**.
    """)
    
    with gr.Accordion("📖 How to Use This Demo", open=False):
        gr.Markdown("""
        ## Features Overview
        
        ### 1. Task Description Prompts
        Add a natural language description of the classification task to guide the model.
        
        **Example:** `"Classify this customer review by sentiment and topic:"`
        
        ---
        
        ### 2. Hierarchical Labels (JSON Format)
        Structure your labels in categories for organized classification:
        
        ```json
        {
            "sentiment": ["positive", "negative", "neutral"],
            "topic": ["product", "service", "shipping"],
            "urgency": ["high", "medium", "low"]
        }
        ```
        
        Or use simple comma-separated labels: `positive, negative, neutral`
        
        ---
        
        ### 3. Few-Shot Examples
        Provide examples to guide the model's understanding:
        
        ```json
        [
            {"text": "Great product, love it!", "labels": ["positive", "product"]},
            {"text": "Shipping was delayed by 2 weeks", "labels": ["negative", "shipping"]}
        ]
        ```
        
        ---
        
        ### 4. Hierarchical Output
        When enabled with hierarchical labels, returns structured scores matching your input format.
        """)
    
    with gr.Accordion("💻 Code Example", open=False):
        gr.Code(
            '''from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

model = GLiClassModel.from_pretrained("knowledgator/gliclass-small-v1")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-small-v1")

pipeline = ZeroShotClassificationPipeline(model, tokenizer, classification_type='multi-label', device='cuda:0')

# Basic usage
text = "The product quality is amazing but delivery was slow"
labels = ["positive", "negative", "product", "shipping"]
results = pipeline(text, labels, threshold=0.5)[0]

# With hierarchical labels
hierarchical_labels = {
    "sentiment": ["positive", "negative", "neutral"],
    "topic": ["product", "service", "shipping"]
}

results = pipeline(
    text, 
    hierarchical_labels,
    prompt="Classify this review:",
    return_hierarchical=True
)[0]

# With few-shot examples
examples = [
    {"text": "Love this item!", "labels": ["sentiment.positive", "topic.product"]},
    {"text": "Terrible customer support", "labels": ["sentiment.negative", "topic.service"]}
]

results = pipeline(
    text,
    hierarchical_labels, 
    examples=examples,
    prompt="Classify customer feedback:"
)[0]
''',
            language="python",
        )
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                value=EXAMPLES[0][0],
                label="📝 Text Input",
                placeholder="Enter the text you want to classify...",
                lines=8
            )
            
            prompt_input = gr.Textbox(
                value=EXAMPLES[0][4],
                label="💡 Task Description Prompt (Optional)",
                placeholder="E.g., 'Classify this customer review by sentiment and topic:'",
                lines=1
            )
        
        with gr.Column(scale=1):
            labels_input = gr.Textbox(
                value=EXAMPLES[0][1],
                label="🏷️ Labels (comma-separated or JSON)",
                placeholder='positive, negative\n\nOR\n\n{"category": ["label1", "label2"]}',
                lines=6
            )
            
            with gr.Row():
                threshold = gr.Slider(
                    0, 1,
                    value=0.5,
                    step=0.01,
                    label="Threshold",
                    info="Confidence threshold for predictions"
                )
            
            with gr.Row():
                multi_label = gr.Checkbox(
                    value=True,
                    label="Multi-label",
                    info="Allow multiple labels per text"
                )
                hierarchical_output = gr.Checkbox(
                    value=False,
                    label="Hierarchical Output",
                    info="Return structured output matching label hierarchy"
                )
            
            with gr.Row():
                output_format = gr.Radio(
                    choices=["visual", "json"],
                    value="visual",
                    label="Output Format",
                    info="Visual: charts/bars | JSON: raw data"
                )
    
    with gr.Accordion("🎯 Few-Shot Examples (Optional)", open=False):
        examples_input = gr.Textbox(
            value="",
            label="Examples (JSON format)",
            placeholder='''[
    {"text": "Example text 1", "labels": ["label1", "label2"]},
    {"text": "Example text 2", "labels": ["label3"]}
]''',
            lines=5
        )
        gr.Markdown("""
        *Provide labeled examples to guide the model. Each example needs a `text` field and a `labels` array.*
        """)
    
    submit_btn = gr.Button("🚀 Classify", variant="primary", size="lg")
    
    output = gr.Label(label="📊 Classification Results")
    output_text = gr.Textbox(
        label="📊 Hierarchical Results", 
        visible=False, 
        lines=10
    )
    output_json = gr.Code(
        label="📊 JSON Output",
        language="json",
        visible=False,
        lines=15
    )
    
    # Dynamic output visibility based on format and hierarchical toggle
    def update_output_visibility(hierarchical: bool, fmt: str):
        if fmt == "json":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
        elif hierarchical:
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    
    hierarchical_output.change(
        fn=update_output_visibility,
        inputs=[hierarchical_output, output_format],
        outputs=[output, output_text, output_json]
    )
    
    output_format.change(
        fn=update_output_visibility,
        inputs=[hierarchical_output, output_format],
        outputs=[output, output_text, output_json]
    )
    
    # Classification function wrapper for different outputs
    def classify_wrapper(text, labels, threshold, multi_label, prompt, examples, hierarchical, fmt):
        result = classification(text, labels, threshold, multi_label, prompt, examples, hierarchical, fmt)
        
        if fmt == "json":
            return None, None, result
        elif hierarchical or isinstance(result, str):
            return None, result, None
        else:
            return result, None, None
    
    # Event handlers
    submit_btn.click(
        fn=classify_wrapper,
        inputs=[input_text, labels_input, threshold, multi_label, prompt_input, examples_input, hierarchical_output, output_format],
        outputs=[output, output_text, output_json]
    )
    
    input_text.submit(
        fn=classify_wrapper,
        inputs=[input_text, labels_input, threshold, multi_label, prompt_input, examples_input, hierarchical_output, output_format],
        outputs=[output, output_text, output_json]
    )
    
    gr.Markdown("### 📚 Example Configurations")
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, labels_input, threshold, multi_label, prompt_input, examples_input, hierarchical_output, output_format],
        outputs=[output, output_text, output_json],
        fn=classify_wrapper,
        cache_examples=False,
        examples_per_page=5
    )
    
    gr.Markdown("""
    ---
    
    ### 🔧 Tips for Best Results
    
    | Feature | Best Practice |
    |---------|---------------|
    | **Prompts** | Be specific about the task, e.g., "Classify by sentiment:" vs "Analyze:" |
    | **Labels** | Use descriptive labels; "customer service issue" > "service" |
    | **Hierarchical** | Group related labels under categories for organized results |
    | **Examples** | 2-3 diverse examples often improve accuracy significantly |
    | **Threshold** | Start at 0.5, lower for more predictions, raise for higher precision |
    """)


if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=True, share=True)