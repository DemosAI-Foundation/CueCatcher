# LLM-Powered Data Analysis in CueCatcher

## Overview

CueCatcher now includes **LLM-powered analysis** that transforms raw behavioral data into compassionate, actionable insights for caregivers and therapists. The system analyzes both **short-horizon** (per-session) and **long-horizon** (longitudinal) data to generate natural language reports.

## Architecture

```
Raw Sensor Data → Three-Tier Aggregation → SQL/JSON Storage → LLM Analyzer → Natural Language Reports
    ↓                    ↓                      ↓                  ↓                ↓
  30fps              Episodes              PostgreSQL          Structured      Caregiver-friendly
 detections        (167ms) + States       or JSON files        prompts         narratives
                   (500ms)                                    with context
```

## How It Works

### 1. Data Collection (Already Implemented)

- **Tier 1**: Per-frame detections (pose, gaze, face, audio) stored in `detections` table
- **Tier 2**: Behavioral episodes (reach, gaze alternation, vocalization bursts) in `episodes` table
- **Tier 3**: Child states (idle, attending, communicating, distressed) tracked over time
- **Tier 4**: AI interpretations with confidence scores in `interpretations` table

### 2. LLM Analysis Module (`server/llm_analyzer.py`)

The new `LLMSessionAnalyzer` class:

#### For Single Sessions:
```python
analyzer = LLMSessionAnalyzer(db_url, session_dir)
await analyzer.connect()
analysis = await analyzer.analyze_session(session_id)
```

Returns:
- `session_summary`: Duration, episode counts, state durations
- `key_moments`: High-confidence interpretations and gaze alternations
- `communication_highlights`: Gaze alternation count, coordinated signals
- `behavioral_patterns`: Most common behaviors and signal combinations
- `llm_prompt`: Pre-formatted prompt ready for LLM

#### For Longitudinal Analysis:
```python
longitudinal = await analyzer.analyze_longitudinal(days=30)
```

Returns:
- `period_summary`: Total sessions, episodes, trends
- `trends`: Increasing/decreasing/stable communication
- `recurring_patterns`: Behaviors seen across multiple sessions
- `max_comm_level_observed`: Highest communication level achieved
- `llm_prompt`: Comprehensive prompt for developmental insights

### 3. REST API Endpoints

#### Single Session Analysis
```bash
GET /api/sessions/{session_id}/analyze
GET /api/sessions/{session_id}/analyze?with_narrative=true
```

Response example:
```json
{
  "session_id": "abc123",
  "session_summary": {
    "duration_minutes": 15.3,
    "total_episodes": 47,
    "gaze_alternation_count": 8
  },
  "key_moments": [
    {
      "type": "episode",
      "episode_type": "gaze_alternation",
      "confidence": 0.89,
      "time_offset": "2025-01-15T10:23:45Z"
    }
  ],
  "llm_prompt": "You are analyzing a CueCatcher session...",
  "video_timestamps": [
    {"timestamp": "10:23:45", "description": "gaze_alternation"}
  ]
}
```

#### Longitudinal Analysis
```bash
GET /api/analysis/longitudinal?days=30
GET /api/analysis/longitudinal?days=90&with_narrative=true
```

Response example:
```json
{
  "period_days": 30,
  "total_sessions": 12,
  "total_episodes": 534,
  "trend": "increasing",
  "max_comm_level_observed": 5,
  "episode_types": {
    "gaze_alternation": 89,
    "reach": 156,
    "vocalization": 203
  },
  "llm_prompt": "You are analyzing 12 CueCatcher sessions over 30 days..."
}
```

### 4. LLM Integration (Ready for Production)

The system is designed to work with any LLM provider:

```python
# Example with OpenAI
import openai

llm_client = openai.AsyncOpenAI(api_key="your-key")
report = await generate_llm_report(analyzer, session_id, llm_client)
```

Supported providers:
- **OpenAI** (GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude 3.5 Sonnet)
- **Ollama** (local Llama 3, Mistral)
- **Azure OpenAI**

### 5. Prompt Engineering

The system uses carefully crafted prompts that:

**For Session Reports:**
- Emphasize strengths-based language
- Explain gaze alternation significance
- Provide specific, actionable strategies
- Avoid jargon ("episode", "interpretation")
- Focus on child's agency and competence

**For Longitudinal Reports:**
- Celebrate developmental progress
- Identify patterns across time
- Suggest evidence-based strategies
- Flag concerns if communication decreases
- Use warm, hopeful language

Example prompt structure:
```
SESSION OVERVIEW:
- Duration: 15.3 minutes
- Total behavioral episodes: 47
- Gaze alternation events: 8

KEY COMMUNICATIVE SIGNALS:
- Coordinated multi-signal behaviors: 5

HIGH-CONFIDENCE MOMENTS:
[...top 10 moments...]

TASK:
Generate a compassionate summary that:
1. Highlights communicative attempts
2. Explains gaze alternation meaning
3. Notes patterns
4. Suggests 1-2 strategies
5. Uses accessible language
```

## Database Requirements

### Minimum (Works Without Database)
- ✅ JSON session files in `/data/sessions/{session_id}/summary.json`
- ✅ Aggregated statistics only
- ⚠️ No per-episode detail for LLM context

### Recommended (PostgreSQL)
```sql
-- Tables used by LLM analyzer:
detections       -- raw per-frame data
episodes         -- behavioral episodes with features
interpretations  -- AI-generated interpretations
sessions         -- session metadata
```

The analyzer gracefully degrades:
- With PostgreSQL: Full episode-level analysis
- With JSON only: Aggregated session-level insights

## Usage Examples

### Example 1: Post-Session Caregiver Report
```python
from server.llm_analyzer import LLMSessionAnalyzer, generate_llm_report
import openai

# After session ends
analyzer = LLMSessionAnalyzer("postgresql://...", Path("/data/sessions"))
await analyzer.connect()

# Generate report with LLM narrative
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
report = await generate_llm_report(analyzer, session_id, client)

# Send to caregiver app
print(report["llm_narrative"])
# "Today your child showed wonderful communication skills! 
# They looked between you and the toy 8 times - this 'gaze 
# alternation' is a sophisticated social signal that shows 
# they're trying to share an experience with you..."
```

### Example 2: Monthly Therapist Report
```python
# Generate longitudinal analysis
longitudinal = await analyzer.analyze_longitudinal(days=30)

# Prepare for LLM
prompt = longitudinal["llm_prompt"]
response = await llm_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a child development specialist."},
        {"role": "user", "content": prompt}
    ]
)

# Include in therapist dashboard
therapist_report = {
    "quantitative_data": longitudinal,
    "narrative_insights": response.choices[0].message.content,
    "recommendations": extract_recommendations(response)
}
```

### Example 3: Real-Time Dashboard Integration
```javascript
// Frontend fetches analysis
const analysis = await fetch('/api/sessions/abc123/analyze');
const data = await analysis.json();

// Display key moments on timeline
data.key_moments.forEach(moment => {
  addMarkerToVideoTimeline(moment.timestamp, moment.description);
});

// Show communication highlights
document.getElementById('gaze-alt-count').textContent = 
  data.communication_highlights.gaze_alternation_count;
```

## Privacy & Safety

### Data Handling
- ✅ All analysis runs **locally** on your server
- ✅ No data sent to external LLM APIs unless explicitly configured
- ✅ Optional local LLM support via Ollama

### LLM Output Safeguards
- Prompts emphasize **hypothesis language** ("may be trying to communicate")
- Reports include **confidence scores** and **alternative interpretations**
- System avoids diagnostic claims (not a medical device)
- Caregiver feedback loop allows correction of misinterpretations

## Performance Considerations

### Context Window Management
- Single session: Top 20 key moments (prevents token overflow)
- Longitudinal: Aggregated statistics + pattern summaries
- Automatic truncation of low-confidence data

### Caching Strategy
```python
# Cache LLM responses to avoid redundant calls
@cache(ttl=3600)  # 1 hour
async def get_cached_analysis(session_id):
    return await analyzer.analyze_session(session_id)
```

### Batch Processing
For clinics with many sessions:
```python
# Process overnight
async def batch_analyze_all_sessions():
    sessions = await get_recent_sessions(hours=24)
    for session in sessions:
        await analyze_session_llm(session.id)
```

## Future Enhancements

### Planned Features
1. **Caregiver Feedback Integration**: Track confirm/reject buttons to improve LLM accuracy
2. **Custom Fine-Tuning**: Train LLM on family-specific communication patterns
3. **Multi-Modal Reports**: Combine LLM narrative with auto-generated charts
4. **Therapist Annotation**: Allow therapists to add notes that inform future LLM analysis
5. **Comparative Analysis**: Compare child's progress against developmental milestones

### Research Opportunities
- Validate LLM-generated insights against therapist assessments
- Study impact of LLM reports on caregiver confidence and responsiveness
- Develop domain-specific LLM fine-tuned on AAC and child development literature

## Files Modified/Created

| File | Purpose |
|------|---------|
| `server/llm_analyzer.py` | Core LLM analysis module |
| `server/api.py` | Added `/analyze` and `/analysis/longitudinal` endpoints |
| `docs/LLM_ANALYSIS_GUIDE.md` | This documentation |

## Testing

### Unit Tests
```bash
pytest tests/test_llm_analyzer.py -v
```

### Integration Test
```bash
# Start server
python -m uvicorn server.main:app

# Test single session analysis
curl http://localhost:8000/api/sessions/{id}/analyze

# Test longitudinal analysis
curl http://localhost:8000/api/analysis/longitudinal?days=30
```

## Conclusion

CueCatcher's LLM analysis transforms complex behavioral data into **compassionate, actionable insights** that empower caregivers and support therapists. The system maintains privacy-first design while leveraging modern AI to make non-verbal communication more visible and understandable.

**Key Benefits:**
- 🎯 Identifies subtle communicative signals caregivers might miss
- 📈 Tracks developmental progress over time
- 💡 Provides evidence-based strategies tailored to the child's patterns
- ❤️ Uses strengths-based, hopeful language
- 🔒 Keeps all data local unless explicitly configured otherwise
