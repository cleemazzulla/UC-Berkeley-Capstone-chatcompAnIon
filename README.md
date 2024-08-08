# chat compAnIon: MIDS Capstone Project Spring 2024

## Overview

**chat compAnIon** is a project dedicated to revolutionizing online safety for children amidst a rapidly evolving digital landscape. Our mission is to empower parents to stay ahead of child grooming and predator detection through cutting-edge technology.

Our team has developed an AI-based solution that integrates seamlessly into gaming chat rooms, swiftly identifying and alerting potential child grooming activities. Our system operates with unmatched speed, analyzing text in real-time and delivering predictions within milliseconds. This ensures that parents or guardians—the first line of defense—are promptly informed and able to take immediate action.

## Team Members

- Leon Gutierrez
- Courtney Mazzulla
- Karsyn Lee
- Julian Rippert
- Sunny Shin
- Raymond Tang

## The Problem

In today's digital age, children's online safety faces unprecedented challenges. The New York Times described this issue as "AN INDUSTRY WITHOUT AN ANSWER," highlighting the significant threats posed by over 500,000 online predators active daily.

### Key Challenges

1. **Lack of Transparency and Data Accessibility:** Online platforms often keep incidents of child grooming confidential, hindering our understanding of the problem and limiting our ability to develop effective prevention strategies.
  
2. **Limited Parental Involvement:** Parents are frequently left unaware of potential threats to their children's safety due to the internal handling of such situations by large corporations.

3. **Urgent Need for Action:** Greater transparency and parental empowerment are essential for timely intervention and protection of children online.

## Our Work

### Advanced Detection System

**chat compAnIon** utilizes an advanced, multi-stage resampling pipeline feeding into a deep learning model. This system navigates the nuances of youth communication and detects even the subtlest signs of predatory behavior.

Due to data scarcity, we employed various sampling techniques, extensive exploratory data analysis (EDA), and tested different probabilistic loss functions to address class imbalance. After rigorous testing, we developed a BERT-CNN-NN model with an F3 score of 0.82, outperforming existing research in this domain.

### Proactive Alert System for Parents

Our alert system is designed to ensure parents remain informed and equipped to take swift action against potential online threats. Key features include:

- **Email Notifications:** Prompt alerts to parents about suspicious activities or grooming attempts.
- **Temporary Chat Room Blocking:** Immediate protective measures by temporarily blocking access to potentially harmful chat rooms.
- **Parental Confirmation:** Ensuring active parental involvement in decision-making regarding their child's safety.
- **Resource Provision and Next Steps:** Providing parents with guidance and resources to handle online threats effectively.

## Collaborative Initiatives

### Open-Sourcing Our Model

To promote transparency and accessibility, we have made our model publicly available on GitHub and Hugging Face. Our goal is to empower developers, researchers, and organizations worldwide to leverage our framework in their efforts to protect children online.

### Exploring Collaboration Opportunities

We are actively engaging with law enforcement, industry leaders, and other key stakeholders to identify synergies and collaborate on strategies to enhance online safety for children.

## Course

**Data Science 210: Capstone, Spring 2024**

## Class Project Gallery

Explore more about this project and others in the Class Project Gallery.

## More Information

- [Our Website](#)
- [GitHub Repository](#)
- [Hugging Face](#)
- [YouTube](#)
"""

# Save the content to a README.md file
with open('/mnt/data/README.md', 'w') as file:
    file.write(readme_content)

"/mnt/data/README.md"
