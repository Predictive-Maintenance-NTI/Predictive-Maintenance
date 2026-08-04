import os
from dotenv import load_dotenv

load_dotenv()


PROVIDER = os.getenv("LLM_PROVIDER", "openai")


FALLBACK_RECOMMENDATIONS = {
    "No Failure": (
        "Machine is operating normally. Continue routine monitoring "
        "and follow the scheduled maintenance plan."
    ),

    "HDF": (
        "Inspect the cooling system and verify proper airflow. "
        "Check for overheating before returning the machine to operation."
    ),

    "PWF": (
        "Inspect the power supply, electrical connections, and motor load. "
        "Verify voltage stability before restarting the machine."
    ),

    "OSF": (
        "Reduce the operating load and inspect the drivetrain for overload "
        "conditions. Review machine settings before restarting."
    ),

    "TWF": (
        "Inspect the cutting tool for excessive wear and replace it if necessary. "
        "Verify machining parameters after replacement."
    ),

    "RNF": (
        "Perform a complete diagnostic inspection to identify unexpected "
        "mechanical or electrical issues before returning the machine to service."
    ),

    "Multiple Failures": (
        "Immediate maintenance is required. Shut down the machine safely "
        "and perform a full inspection of critical systems before restarting."
    ),
}


def build_prompt(prediction, probability, machine_data):

    reading_summary = (
        f"Machine type: {machine_data['machine_type']}\n"
        f"Air temperature: {machine_data['air_temperature']} K\n"
        f"Process temperature: {machine_data['process_temperature']} K\n"
        f"Rotational speed: {machine_data['rotational_speed']} rpm\n"
        f"Torque: {machine_data['torque']} Nm\n"
        f"Tool wear: {machine_data['tool_wear']} min\n"
        f"Model prediction: {prediction} "
        f"(confidence {probability:.1%})\n"
    )

    if prediction == "No Failure":
        task = (
            "The machine is operating normally. "
            "Provide short preventive maintenance advice."
        )

    else:
        task = (
            f"The machine is predicted to have {prediction} failure. "
            "Explain possible cause, immediate action, "
            "maintenance recommendation, and prevention tips."
        )

    return (
        reading_summary
        + "\n"
        + task
        + "\n\nKeep the response concise under 150 words."
    )


def call_openai(prompt):

    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing."
        )

    client = OpenAI(
        api_key=api_key
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content



PROVIDERS = {
    "openai": call_openai,
}



def get_recommendation(prediction, probability, machine_data):

    prompt = build_prompt(
        prediction,
        probability,
        machine_data
    )

    call_fn = PROVIDERS.get(PROVIDER, call_openai)

    try:
        return call_fn(prompt)

    except Exception as e:

        print("LLM Error:", e)

        return FALLBACK_RECOMMENDATIONS.get(
            prediction,
            "No recommendation available."
        )