import express from "express";
import dotenv from "dotenv";
import twilio from "twilio";
import { GoogleGenAI } from "@google/genai";
import wav from "wav";
import fs from "fs";
import path from "path";
import crypto from "crypto";
import { fileURLToPath } from "url";

dotenv.config();

const {
  PORT = 3000,
  BASE_URL,
  GEMINI_API_KEY,
  TWILIO_AUTH_TOKEN,
} = process.env;

if (!BASE_URL || !GEMINI_API_KEY || !TWILIO_AUTH_TOKEN) {
  throw new Error(
    "Missing BASE_URL, GEMINI_API_KEY, or TWILIO_AUTH_TOKEN in your .env file."
  );
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use("/audio", express.static(path.join(__dirname, "public", "audio")));

const sessions = new Map();
const audioDir = path.join(__dirname, "public", "audio");
fs.mkdirSync(audioDir, { recursive: true });

const SYSTEM_PROMPT = `
You are CareRing, a warm, patient, elderly-friendly phone companion.

Your role on the phone:
- Speak simply, slowly, and clearly.
- Keep each reply short: usually 1 to 3 sentences.
- Ask only one question at a time.
- Sound calm, respectful, and human.
- Avoid slang, jargon, or overly robotic wording.
- If the caller seems confused, gently repeat or rephrase.
- Always acknowledge what they said before moving on.
- Help with check-ins like mood, meals, hydration, medication, pets, safety, and loneliness.
- Never overwhelm the caller with long lists.
- If they sound distressed, very unwell, unsafe, or say they fell / cannot breathe / have chest pain / are in danger, tell them to call emergency services or a trusted caretaker immediately.
- If they request their caretaker, respond supportively and say you will notify the caretaker system.

Conversation style rules:
- Use reassuring phrases like “Take your time,” “That’s okay,” and “I’m here with you.”
- Prefer yes/no or very easy questions.
- End with one gentle next question.
- Do not mention policies or that you are following instructions.
- Never say you are replacing medical professionals.

When summarizing the user's state internally, pay attention to:
- mood
- medication taken or missed
- food/water
- pets/tasks completed
- loneliness or desire to speak to caretaker
- any risk flags
`;

function getSession(callSid) {
  if (!sessions.has(callSid)) {
    sessions.set(callSid, {
      history: [],
      summary: {
        mood: null,
        medication: null,
        meals: null,
        hydration: null,
        pets: null,
        loneliness: null,
        riskFlags: [],
      },
    });
  }
  return sessions.get(callSid);
}

function updateSummaryFromText(session, text) {
  const lower = text.toLowerCase();

  if (["sad", "lonely", "upset", "worried", "scared"].some((w) => lower.includes(w))) {
    session.summary.mood = text;
  }
  if (lower.includes("medicine") || lower.includes("medication") || lower.includes("pill")) {
    session.summary.medication = text;
  }
  if (lower.includes("ate") || lower.includes("meal") || lower.includes("breakfast") || lower.includes("lunch") || lower.includes("dinner")) {
    session.summary.meals = text;
  }
  if (lower.includes("water") || lower.includes("drink") || lower.includes("hydrated")) {
    session.summary.hydration = text;
  }
  if (lower.includes("cat") || lower.includes("dog") || lower.includes("pet") || lower.includes("fed")) {
    session.summary.pets = text;
  }
  if (lower.includes("lonely") || lower.includes("call my daughter") || lower.includes("call my son") || lower.includes("caretaker")) {
    session.summary.loneliness = text;
  }

  const riskWords = [
    "fell",
    "fall",
    "dizzy",
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "help me",
    "emergency",
    "bleeding",
    "confused",
    "not safe",
  ];

  for (const word of riskWords) {
    if (lower.includes(word) && !session.summary.riskFlags.includes(word)) {
      session.summary.riskFlags.push(word);
    }
  }
}

async function generateReply(callSid, userText) {
  const session = getSession(callSid);

  session.history.push({ role: "user", text: userText });
  updateSummaryFromText(session, userText);

  const conversation = session.history
    .slice(-12)
    .map((m) => `${m.role === "user" ? "Caller" : "CareRing"}: ${m.text}`)
    .join("\n");

  const prompt = `${SYSTEM_PROMPT}

Current lightweight summary:
${JSON.stringify(session.summary, null, 2)}

Recent conversation:
${conversation}

Write the next phone response for the elderly caller.
Requirements:
- maximum 60 words
- warm and very easy to understand
- only one question
- mention emergency help only if truly needed
- no bullet points
`;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [{ parts: [{ text: prompt }] }],
  });

  const text = response.text?.trim() || "Hello, I’m here with you. Have you had some water today?";
  session.history.push({ role: "assistant", text });
  return text;
}

async function synthesizeSpeech(text) {
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-preview-tts",
    contents: [
      {
        parts: [
          {
            text: `Say this warmly, slowly, clearly, and in a comforting tone for an elderly listener: ${text}`,
          },
        ],
      },
    ],
    config: {
      responseModalities: ["AUDIO"],
      speechConfig: {
        voiceConfig: {
          prebuiltVoiceConfig: { voiceName: "Kore" },
        },
      },
    },
  });

  const data = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
  if (!data) throw new Error("No audio returned from Gemini TTS.");

  const audioBuffer = Buffer.from(data, "base64");
  const filename = `${crypto.randomUUID()}.wav`;
  const filepath = path.join(audioDir, filename);

  await saveWaveFile(filepath, audioBuffer);
  return `${BASE_URL}/audio/${filename}`;
}

async function saveWaveFile(filename, pcmData, channels = 1, rate = 24000, sampleWidth = 2) {
  return new Promise((resolve, reject) => {
    const writer = new wav.FileWriter(filename, {
      channels,
      sampleRate: rate,
      bitDepth: sampleWidth * 8,
    });

    writer.on("finish", resolve);
    writer.on("error", reject);
    writer.write(pcmData);
    writer.end();
  });
}

function isRequestFromTwilio(req) {
  const signature = req.header("X-Twilio-Signature");
  if (!signature) return false;

  const url = `${BASE_URL}${req.originalUrl}`;
  return twilio.validateRequest(TWILIO_AUTH_TOKEN, signature, url, req.body);
}

app.post("/voice", async (req, res) => {
  if (!isRequestFromTwilio(req)) {
    return res.status(403).send("Invalid Twilio signature");
  }

  const twiml = new twilio.twiml.VoiceResponse();
  const callSid = req.body.CallSid;
  const firstMessage = "Hello, this is CareRing checking in. Take your time. How are you feeling today?";

  try {
    const audioUrl = await synthesizeSpeech(firstMessage);
    twiml.play(audioUrl);

    const gather = twiml.gather({
      input: "speech",
      action: "/gather",
      method: "POST",
      speechTimeout: "auto",
      speechModel: "phone_call",
      enhanced: true,
      language: "en-US",
      actionOnEmptyResult: true,
    });

    gather.pause({ length: 1 });
    getSession(callSid);

    res.type("text/xml").send(twiml.toString());
  } catch (error) {
    console.error(error);
    twiml.say("Hello. I am calling to check in. How are you feeling today?");
    twiml.redirect({ method: "POST" }, "/voice");
    res.type("text/xml").send(twiml.toString());
  }
});

app.post("/gather", async (req, res) => {
  if (!isRequestFromTwilio(req)) {
    return res.status(403).send("Invalid Twilio signature");
  }

  const twiml = new twilio.twiml.VoiceResponse();
  const callSid = req.body.CallSid;
  const transcript = (req.body.SpeechResult || "").trim();

  try {
    let reply;

    if (!transcript) {
      reply = "That’s okay. Take your time. Could you please say that once more?";
    } else {
      reply = await generateReply(callSid, transcript);
    }

    const audioUrl = await synthesizeSpeech(reply);
    twiml.play(audioUrl);

    const gather = twiml.gather({
      input: "speech",
      action: "/gather",
      method: "POST",
      speechTimeout: "auto",
      speechModel: "phone_call",
      enhanced: true,
      language: "en-US",
      actionOnEmptyResult: true,
    });

    gather.pause({ length: 1 });
    res.type("text/xml").send(twiml.toString());
  } catch (error) {
    console.error(error);
    twiml.say("I’m sorry, I had a little trouble hearing that. Could you say it again?");
    twiml.redirect({ method: "POST" }, "/gather");
    res.type("text/xml").send(twiml.toString());
  }
});

app.post("/call-summary", (req, res) => {
  if (!isRequestFromTwilio(req)) {
    return res.status(403).send("Invalid Twilio signature");
  }

  const callSid = req.body.CallSid;
  const session = sessions.get(callSid);
  if (!session) {
    return res.json({ ok: true, summary: null });
  }

  res.json({
    ok: true,
    summary: session.summary,
    transcript: session.history,
  });
});

app.get("/health", (_req, res) => {
  res.json({ ok: true });
});

app.listen(PORT, () => {
  console.log(`CareRing voicebot running on port ${PORT}`);
});
