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
  NODE_ENV = "development",
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

const audioDir = path.join(__dirname, "public", "audio");
fs.mkdirSync(audioDir, { recursive: true });
app.use("/audio", express.static(audioDir));

const sessions = new Map();

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

When summarizing the caller's state internally, pay attention to:
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
      createdAt: new Date().toISOString(),
    });
  }

  return sessions.get(callSid);
}

function updateSummaryFromText(session, text) {
  const lower = text.toLowerCase();

  if (["sad", "lonely", "upset", "worried", "scared", "tired", "happy", "okay", "fine"].some((w) => lower.includes(w))) {
    session.summary.mood = text;
  }

  if (["medicine", "medication", "pill", "tablet", "dose"].some((w) => lower.includes(w))) {
    session.summary.medication = text;
  }

  if (["ate", "eating", "meal", "breakfast", "lunch", "dinner", "food"].some((w) => lower.includes(w))) {
    session.summary.meals = text;
  }

  if (["water", "drink", "drank", "hydrated", "tea", "juice"].some((w) => lower.includes(w))) {
    session.summary.hydration = text;
  }

  if (["cat", "dog", "pet", "fed", "litter"].some((w) => lower.includes(w))) {
    session.summary.pets = text;
  }

  if (["lonely", "caretaker", "daughter", "son", "call someone", "talk to someone"].some((w) => lower.includes(w))) {
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
    "hurt",
    "injured",
  ];

  for (const word of riskWords) {
    if (lower.includes(word) && !session.summary.riskFlags.includes(word)) {
      session.summary.riskFlags.push(word);
    }
  }
}

async function generateReply(callSid, userText) {
  const session = getSession(callSid);

  session.history.push({ role: "user", text: userText, at: new Date().toISOString() });
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
  session.history.push({ role: "assistant", text, at: new Date().toISOString() });

  return text;
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
  if (!data) {
    throw new Error("No audio returned from Gemini TTS.");
  }

  const audioBuffer = Buffer.from(data, "base64");
  const filename = `${crypto.randomUUID()}.wav`;
  const filepath = path.join(audioDir, filename);

  await saveWaveFile(filepath, audioBuffer);
  return `${BASE_URL}/audio/${filename}`;
}

function isRequestFromTwilio(req) {
  if (NODE_ENV !== "production") {
    return true;
  }

  const signature = req.header("X-Twilio-Signature");
  if (!signature) {
    return false;
  }

  const url = `${BASE_URL}${req.originalUrl}`;
  return twilio.validateRequest(TWILIO_AUTH_TOKEN, signature, url, req.body);
}

function buildGather(twiml) {
  const gather = twiml.gather({
    input: "speech",
    action: `${BASE_URL}/gather`,
    method: "POST",
    speechTimeout: "auto",
    speechModel: "phone_call",
    enhanced: true,
    language: "en-US",
    actionOnEmptyResult: true,
  });

  gather.pause({ length: 1 });
  return gather;
}

app.get("/health", (_req, res) => {
  res.json({ ok: true, environment: NODE_ENV });
});

app.get("/test-ai", async (_req, res) => {
  try {
    const reply = await generateReply(
      "test-call",
      "I am feeling a little lonely today and I forgot to drink water."
    );

    res.json({ ok: true, reply });
  } catch (error) {
    console.error("Error in /test-ai:", error);
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.get("/test-tts", async (_req, res) => {
  try {
    const audioUrl = await synthesizeSpeech(
      "Hello, this is CareRing checking in. Have you had some water today?"
    );

    res.json({ ok: true, audioUrl });
  } catch (error) {
    console.error("Error in /test-tts:", error);
    res.status(500).json({ ok: false, error: error.message });
  }
});

app.post("/voice", async (req, res) => {
  console.log("VOICE HIT");
  console.log("Body:", req.body);

  if (!isRequestFromTwilio(req)) {
    console.log("Twilio signature failed on /voice");
    return res.status(403).send("Invalid Twilio signature");
  }

  const twiml = new twilio.twiml.VoiceResponse();
  const callSid = req.body.CallSid || crypto.randomUUID();
  getSession(callSid);

  try {
    const firstMessage = "Hello, this is CareRing checking in. Take your time. How are you feeling today?";
    const audioUrl = await synthesizeSpeech(firstMessage);

    console.log("Initial audio URL:", audioUrl);

    twiml.play(audioUrl);
    buildGather(twiml);

    res.type("text/xml").send(twiml.toString());
  } catch (error) {
    console.error("Error in /voice:", error);

    twiml.say("Hello, this is CareRing checking in. Take your time. How are you feeling today?");
    buildGather(twiml);

    res.type("text/xml").send(twiml.toString());
  }
});

app.post("/gather", async (req, res) => {
  console.log("GATHER HIT");
  console.log("Body:", req.body);

  if (!isRequestFromTwilio(req)) {
    console.log("Twilio signature failed on /gather");
    return res.status(403).send("Invalid Twilio signature");
  }

  const twiml = new twilio.twiml.VoiceResponse();
  const callSid = req.body.CallSid || "unknown-call";
  const transcript = (req.body.SpeechResult || "").trim();

  console.log("Transcript:", transcript);

  try {
    let reply;

    if (!transcript) {
      reply = "That’s okay. Take your time. Could you please say that once more?";
    } else {
      reply = await generateReply(callSid, transcript);
    }

    console.log("AI reply:", reply);

    let audioUrl;
    try {
      audioUrl = await synthesizeSpeech(reply);
      console.log("Audio URL:", audioUrl);
      twiml.play(audioUrl);
    } catch (ttsError) {
      console.error("TTS failed in /gather:", ttsError);
      twiml.say(reply);
    }

    buildGather(twiml);
    res.type("text/xml").send(twiml.toString());
  } catch (error) {
    console.error("Error in /gather:", error);

    twiml.say("I'm sorry, something went wrong. Could you say that again?");
    buildGather(twiml);

    res.type("text/xml").send(twiml.toString());
  }
});

app.post("/call-summary", (req, res) => {
  console.log("CALL SUMMARY HIT");
  console.log("Body:", req.body);

  if (!isRequestFromTwilio(req)) {
    console.log("Twilio signature failed on /call-summary");
    return res.status(403).send("Invalid Twilio signature");
  }

  const callSid = req.body.CallSid;
  const session = callSid ? sessions.get(callSid) : null;

  res.json({
    ok: true,
    summary: session?.summary || null,
    transcript: session?.history || [],
  });
});

app.listen(PORT, () => {
  console.log(`CareRing voicebot running on port ${PORT}`);
  console.log(`Base URL: ${BASE_URL}`);
  console.log(`Mode: ${NODE_ENV}`);
});
