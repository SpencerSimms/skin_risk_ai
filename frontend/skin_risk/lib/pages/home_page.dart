import 'package:flutter/material.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const SizedBox(height: 40),
          // App title
          const Text(
            'Welcome to Skin Risk AI',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
          ),

          const SizedBox(height: 10),
          const Text(
            'Detect potential skin cancer risks early using AI-powered analysis. '
            'Upload or scan an image to get started.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 16,
              color: Colors.black54,
              height: 1.5,
            ),
          ),

          const SizedBox(height: 30),

          // Image or icon
          Icon(
            Icons.health_and_safety,
            color: Theme.of(context).colorScheme.primary,
            size: 100,
          ),

          const SizedBox(height: 30),

          // "Start Scan" button
          ElevatedButton.icon(
            onPressed: () {
              // Navigate to scan page
              Navigator.pushNamed(context, '/scan');
            },
            icon: const Icon(Icons.camera_alt),
            label: const Text('Start a Scan'),
            style: ElevatedButton.styleFrom(
              padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              textStyle: const TextStyle(fontSize: 18),
            ),
          ),

          const SizedBox(height: 50),

          // Section: How it works
          const Text(
            'How It Works',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 20),

          const ListTile(
            leading: Icon(Icons.image_search, color: Colors.deepPurple),
            title: Text('1. Take or upload a photo of your skin.'),
          ),
          const ListTile(
            leading: Icon(Icons.insights, color: Colors.deepPurple),
            title: Text('2. Our AI model analyzes the image for potential risks.'),
          ),
          const ListTile(
            leading: Icon(Icons.assignment_turned_in, color: Colors.deepPurple),
            title: Text('3. View your risk level and personalized suggestions.'),
          ),

          const SizedBox(height: 40),
          const Text(
            'Disclaimer: This tool does not replace professional medical advice.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.black45, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
