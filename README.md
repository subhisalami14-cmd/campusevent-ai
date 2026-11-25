# CampusEvent AI – אבן דרך 1

**מגישים:** ראיד אבו דבוס, סובחי סלאמה | קורס סוכני AI, המכללה האקדמית עמק יזרעאל, תשפ"ז סמסטר א'

סוכן לתיאום חדרים, ציוד אורקולי **ובקרת תקציב** לאירועי קמפוס. אבן הדרך הזו מראה את היסודות: כלים עם חוזים, אימות והרשאות; טיפול בכשל זמני; מצב נפרד לכל שיחה; ולולאת סוכן עם עקבות.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/subhisalami14-cmd/campusevent-ai/blob/main/notebooks/campusevent_ai_milestone1.ipynb)

## הרצה ב-Google Colab (הדרך הנבדקת)
1. פותחים את המחברת דרך הכפתור למעלה.
2. מריצים את כל התאים לפי הסדר (Runtime → Run all). המחברת משכפלת את המאגר, מתקינה את `requirements.txt`, מריצה את הבדיקות ואת ההדגמה.
3. ברירת המחדל היא **מודל מדומה**, בלי מפתח. למודל אמיתי: מוסיפים `GEMINI_API_KEY` ב-Colab Secrets ומריצים את התא האחרון.

## הרצה מקומית
```bash
git clone https://github.com/subhisalami14-cmd/campusevent-ai.git
cd campusevent-ai
pip install -r requirements.txt
pytest                 # כל הבדיקות, פקודה אחת, בלי רשת ובלי מפתח
python -m src.demo     # הרצה תקינה, כשל זמני והשוואת הקשר; שומר עקבות ב-docs/traces
```

## מבנה המאגר
```
src/
  tools.py      הכלים, החוזים, אימות, הרשאות, timeout וניסיונות חוזרים
  state.py      AgentState, apply_event, StateStore, NoteStore, build_context
  agent.py      לולאת הסוכן, MockModel (מדומה), GeminiModel (אופציונלי)
  demo.py       הדגמות ויצירת קבצי העקבות
  data/         נתונים סינתטיים (15 חדרים, מלאי ציוד, יומן התחלתי)
tests/          pytest – לפחות 3 בדיקות לכל כלי + הפרדת שיחות + לולאה
notebooks/      מחברת הקולאב
docs/
  milestone1.md מסמך התיעוד
  ai_usage.md   נספח שימוש ב-AI
  traces/       קבצי העקבות (JSONL)
```

## רכיבים מדומים (מסומנים במפורש)
| רכיב | מה הוא מדמה | איפה |
|---|---|---|
| `CampusDB` | שירות יומן הקמפוס ומלאי הציוד, כולל הזרקת כשלים (503, timeout, כשל קבוע) | `src/tools.py` |
| `MockModel` | מודל שפה – מבוסס כללים, מחליט רק לפי ההקשר שהועבר אליו | `src/agent.py` |

כל הנתונים סינתטיים. אין במאגר מפתחות API, סיסמאות או מידע אישי.
