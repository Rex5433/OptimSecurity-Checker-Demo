const CLASS_NAMES = ["Very Weak", "Weak", "Average", "Strong", "Very Strong"];
const SCORE_VALUES = [0, 25, 50, 75, 100];
const substitutions = {"@":"a","4":"a","3":"e","1":"i","!":"i","0":"o","$":"s","5":"s","7":"t","+":"t","2":"z","8":"b"};
let session = null;
let charToIndex = {};

const form = document.getElementById("checker-form");
const input = document.getElementById("password");
const count = document.getElementById("password-count");
const errorBox = document.getElementById("checker-error");

input.addEventListener("input", () => { count.textContent = input.value.length; });

async function loadChecker() {
    try {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
        const [characterMap, model] = await Promise.all([
            fetch("model/char_to_idx.json").then(response => response.json()),
            ort.InferenceSession.create("model/password_model.onnx", { 
                executionProviders: ["wasm"],
                externalData: [{
                    path: "password_model.onnx.data",
                    data: "model/password_model.onnx.data"
                }]
            } )
        ]);
        charToIndex = characterMap;
        session = model;
    } catch (error) {
        showError("The CNN model could not be loaded. Run the page through GitHub Pages or a local web server.");
    }
}

function findPolicyReasons(password) {
    const reasons = [];
    const lower = password.toLowerCase();
    const normalized = [...lower].map(character => substitutions[character] || character).join("");
    const badWords = ["password","qwerty","admin","letmein","welcome","iloveyou","abc123","football","monkey","dragon"];
    const commonWords = ["fish","cat","dog","love","user","login","home","school","test","admin","welcome","dragon","monkey"];
    const sequences = ["0123456789","1234567890","abcdefghijklmnopqrstuvwxyz","qwertyuiop","asdfghjkl","zxcvbnm"];

    if (password.length < 10) reasons.push("too_short");
    if (badWords.some(word => lower.includes(word))) reasons.push("common_bad_substring");
    if (/(19\d{2}|20\d{2}|21\d{2})$/.test(password)) reasons.push("ends_with_year");
    if (/(.)\1{3,}/.test(password)) reasons.push("repeated_characters");
    if (sequences.some(sequence => [...Array(Math.max(0, sequence.length - 3))].some((_, index) => lower.includes(sequence.slice(index, index + 4))))) reasons.push("sequential_pattern");
    if (/^[A-Za-z]+$/.test(password)) reasons.push("too_much_plaintext");
    if (new Set(password).size < Math.max(4, password.length * 0.45)) reasons.push("low_uniqueness");
    if (commonWords.some(word => word.length >= 4 && normalized.includes(word))) reasons.push("dictionary_root_with_common_substitution");
    if (/[A-Za-z]{5,}[\d!@#$%^&*]{1,4}$/.test(password)) reasons.push("word_plus_predictable_suffix");
    if (/^[\W\d]{0,2}[A-Za-z]{5,}[\W\d]{0,2}$/.test(password)) reasons.push("edge_only_changes");
    if (/^[A-Za-z]{4,}[\d!@#$%^&*]{1,4}$/.test(password)) reasons.push("word_plus_suffix");
    if ([...password].some(character => substitutions[character]) && /[a-z]{4,}/.test(normalized)) reasons.push("common_character_substitutions");
    return [...new Set(reasons)];
}

function calculatePenalty(reasons, label, confidence) {
    const weights = {too_short:32,common_bad_substring:32,sequential_pattern:20,too_much_plaintext:20,dictionary_root_with_common_substitution:16,word_plus_predictable_suffix:14,word_plus_suffix:12,edge_only_changes:12,ends_with_year:8,common_character_substitutions:6,low_uniqueness:6,repeated_characters:8};
    const settings = {"Very Strong":[0.55,22],"Strong":[0.70,26],"Average":[0.85,30]};
    let [multiplier, cap] = settings[label] || [1, 35];
    if ((label === "Strong" || label === "Very Strong") && confidence >= 0.9) multiplier *= 0.9;
    return Math.min(reasons.reduce((total, reason) => total + (weights[reason] || 0), 0) * multiplier, cap);
}

async function checkPassword(password) {
    const encoded = [...password].map(character => BigInt(charToIndex[character] || 0));
    const data = BigInt64Array.from(encoded.length ? encoded : [0n]);
    const tensor = new ort.Tensor("int64", data, [1, data.length]);
    const outputs = await session.run({ [session.inputNames[0]]: tensor });
    const logits = Array.from(outputs[session.outputNames[0]].data);
    const maximum = Math.max(...logits);
    const exponentials = logits.map(value => Math.exp(value - maximum));
    const total = exponentials.reduce((sum, value) => sum + value, 0);
    const probabilities = exponentials.map(value => value / total);
    const predictedIndex = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[predictedIndex];
    const cnnScore = probabilities.reduce((score, probability, index) => score + probability * SCORE_VALUES[index], 0);
    const finalScore = Math.max(0, Math.min(100, cnnScore - calculatePenalty(findPolicyReasons(password), CLASS_NAMES[predictedIndex], confidence)));
    return CLASS_NAMES[Math.max(0, Math.min(4, Math.round(finalScore / 25)))];
}

form.addEventListener("submit", async event => {
    event.preventDefault();
    const password = input.value.trim();
    if (!password) return showError("Please enter a password.");
    if (!session) return showError("The CNN model is still loading. Please try again.");
    try {
        hideError();
        showResult(password, await checkPassword(password));
    } catch (error) {
        showError("The password could not be checked. Please try again.");
    }
});

function showResult(password, strength) {
    document.getElementById("checker-empty").hidden = true;
    document.getElementById("checker-result").hidden = false;
    document.getElementById("result-password").textContent = password;
    document.getElementById("result-strength").textContent = strength;
    document.getElementById("result-length").textContent = password.length;
    const pill = document.getElementById("result-pill");
    pill.textContent = strength;
    pill.className = "checker-pill " + (strength === "Average" ? "medium" : (strength === "Strong" || strength === "Very Strong" ? "strong" : "weak"));
}

function showError(message) { errorBox.textContent = message; errorBox.hidden = false; }
function hideError() { errorBox.hidden = true; }
loadChecker();
