"""Treatment recommendations for plant diseases."""

# Treatment recommendations based on disease
TREATMENTS = {
    # Apple diseases
    "Apple___Apple_scab": {
        "symptoms": "Dark, olive-green spots on leaves that turn brown and may cause leaves to drop.",
        "treatment": "Apply fungicides containing captan or myclobutanil. Remove and destroy infected leaves. "
        "Ensure good air circulation around trees.",
    },
    "Apple___Black_rot": {
        "symptoms": "Circular brown spots with concentric rings on leaves and fruit.",
        "treatment": "Prune out dead wood and infected branches. Apply fungicides during the growing season. "
        "Remove mummified fruits.",
    },
    "Apple___Cedar_apple_rust": {
        "symptoms": "Bright orange-red spots on leaves with yellow halos.",
        "treatment": "Apply fungicides in early spring. Remove nearby juniper/cedar trees if possible. "
        "Use resistant apple varieties.",
    },
    "Apple___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Continue regular maintenance: proper watering, fertilization, and pruning.",
    },
    # Blueberry
    "Blueberry___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Maintain proper soil pH (4.5-5.5), ensure good drainage, and regular fertilization.",
    },
    # Cherry diseases
    "Cherry_(including_sour)___Powdery_mildew": {
        "symptoms": "White powdery coating on leaves and young shoots.",
        "treatment": "Apply sulfur-based or horticultural oil fungicides. Prune for better air circulation. "
        "Remove infected plant debris.",
    },
    "Cherry_(including_sour)___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Continue regular care: proper watering, annual pruning, and monitoring for pests.",
    },
    # Corn diseases
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "symptoms": "Rectangular gray or tan lesions on leaves.",
        "treatment": "Use resistant hybrids. Apply foliar fungicides if severe. "
        "Practice crop rotation and tillage to reduce inoculum.",
    },
    "Corn_(maize)___Common_rust_": {
        "symptoms": "Small, circular to elongate brown pustules on leaves.",
        "treatment": "Plant resistant varieties. Apply fungicides if infection occurs early. "
        "Ensure proper plant spacing.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "symptoms": "Long, elliptical, grayish-green or tan lesions on leaves.",
        "treatment": "Use resistant hybrids. Apply fungicides preventively. Remove crop debris after harvest.",
    },
    "Corn_(maize)___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Maintain proper fertilization, irrigation, and monitor for early signs of disease.",
    },
    # Grape diseases
    "Grape___Black_rot": {
        "symptoms": "Circular, tan spots with dark margins on leaves; black, shriveled berries.",
        "treatment": "Apply fungicides from bud break through veraison. "
        "Remove mummified berries and infected plant material.",
    },
    "Grape___Esca_(Black_Measles)": {
        "symptoms": "Dark streaks in wood, leaf discoloration, and shriveled berries.",
        "treatment": "No cure available. Remove severely infected vines. "
        "Avoid pruning during wet weather. Paint pruning wounds.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "symptoms": "Angular, yellowish spots that turn brown on leaves.",
        "treatment": "Apply copper-based fungicides. Improve air circulation. Remove infected leaves.",
    },
    "Grape___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Continue regular vineyard management: pruning, training, and disease monitoring.",
    },
    # Orange diseases
    "Orange___Haunglongbing_(Citrus_greening)": {
        "symptoms": "Yellowing of leaves, lopsided fruit, and tree decline.",
        "treatment": "No cure available. Remove infected trees. Control Asian citrus psyllid vector. "
        "Use certified disease-free plants.",
    },
    # Peach diseases
    "Peach___Bacterial_spot": {
        "symptoms": "Small, dark spots on leaves and fruit with yellow halos.",
        "treatment": "Apply copper-based bactericides. Use resistant varieties. Avoid overhead irrigation.",
    },
    "Peach___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Maintain regular care: dormant oil sprays, proper pruning, and thinning.",
    },
    # Pepper diseases
    "Pepper,_bell___Bacterial_spot": {
        "symptoms": "Dark, water-soaked spots on leaves and fruit.",
        "treatment": "Use disease-free seeds and transplants. Apply copper-based sprays. Practice crop rotation.",
    },
    "Pepper,_bell___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Ensure proper spacing, avoid overhead watering, and maintain soil health.",
    },
    # Potato diseases
    "Potato___Early_blight": {
        "symptoms": "Dark, concentric spots on lower leaves that spread upward.",
        "treatment": "Apply fungicides containing chlorothalonil or mancozeb. Practice crop rotation. "
        "Remove plant debris.",
    },
    "Potato___Late_blight": {
        "symptoms": "Water-soaked spots that turn brown and spread rapidly.",
        "treatment": "Apply fungicides preventively. Use resistant varieties. "
        "Destroy infected plants immediately.",
    },
    "Potato___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Continue proper hilling, regular watering, and monitoring for pests and diseases.",
    },
    # Raspberry
    "Raspberry___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Maintain proper pruning, mulching, and ensure good air circulation.",
    },
    # Soybean
    "Soybean___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Practice crop rotation, use quality seeds, and monitor for pests.",
    },
    # Squash diseases
    "Squash___Powdery_mildew": {
        "symptoms": "White, powdery coating on leaves and stems.",
        "treatment": "Apply sulfur or potassium bicarbonate fungicides. Ensure good air circulation. "
        "Remove infected leaves.",
    },
    # Strawberry diseases
    "Strawberry___Leaf_scorch": {
        "symptoms": "Purple or brown spots on leaves with purple margins.",
        "treatment": "Remove infected leaves. Apply fungicides. "
        "Ensure proper plant spacing and avoid overhead watering.",
    },
    "Strawberry___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Maintain proper mulching, regular feeding, and renovation after harvest.",
    },
    # Tomato diseases
    "Tomato___Bacterial_spot": {
        "symptoms": "Dark, greasy spots on leaves and raised spots on fruit.",
        "treatment": "Use disease-free seeds. Apply copper-based bactericides. Avoid overhead irrigation.",
    },
    "Tomato___Early_blight": {
        "symptoms": "Brown spots with concentric rings on lower leaves.",
        "treatment": "Apply fungicides. Practice crop rotation. Stake plants and mulch to prevent soil splash.",
    },
    "Tomato___Late_blight": {
        "symptoms": "Large, irregular water-soaked spots that turn brown.",
        "treatment": "Apply fungicides preventively. Remove infected plants. Avoid overhead watering.",
    },
    "Tomato___Leaf_Mold": {
        "symptoms": "Yellow spots on upper leaf surface, olive-green mold beneath.",
        "treatment": "Improve ventilation in greenhouses. Apply fungicides. Use resistant varieties.",
    },
    "Tomato___Septoria_leaf_spot": {
        "symptoms": "Small, circular spots with dark borders and gray centers.",
        "treatment": "Remove infected leaves. Apply fungicides. Avoid overhead watering. Practice crop rotation.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "symptoms": "Fine webbing on leaves, yellow stippling, and leaf bronzing.",
        "treatment": "Apply miticides or insecticidal soaps. Release predatory mites. Keep plants well-watered.",
    },
    "Tomato___Target_Spot": {
        "symptoms": "Brown spots with concentric rings on leaves and fruit.",
        "treatment": "Apply fungicides. Remove infected plant debris. Ensure good air circulation.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "symptoms": "Yellowing and upward curling of leaves, stunted growth.",
        "treatment": "Control whitefly vectors. Use resistant varieties. Remove infected plants.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "symptoms": "Mottled light and dark green patterns on leaves, distorted growth.",
        "treatment": "Use virus-free seeds. Disinfect tools. Remove infected plants. Control aphids.",
    },
    "Tomato___healthy": {
        "symptoms": "No disease detected.",
        "treatment": "Continue proper care: regular watering, pruning suckers, and monitoring for pests.",
    },
}


def get_treatment_recommendation(disease_name: str) -> str:
    """Get treatment recommendation for a disease.

    Args:
        disease_name: Name of the disease

    Returns:
        Treatment recommendation string
    """
    if disease_name in TREATMENTS:
        info = TREATMENTS[disease_name]
        return f"Symptoms: {info['symptoms']}\n\nTreatment: {info['treatment']}"
    else:
        return (
            "Treatment information not available for this disease. "
            "Please consult with a local agricultural expert."
        )


def get_all_diseases() -> list:
    """Get list of all diseases in the database."""
    return list(TREATMENTS.keys())


def get_disease_info(disease_name: str) -> dict:
    """Get complete information about a disease."""
    return TREATMENTS.get(
        disease_name,
        {
            "symptoms": "Information not available",
            "treatment": "Please consult with a local agricultural expert",
        },
    )
