import 'dart:io';
import 'package:flutter/material.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  // Mock data for now — replace with real API or local storage later
  List<Map<String, dynamic>> scanHistory = [
    {
      'imagePath': 'assets/sample1.jpg',
      'label': 'melanoma',
      'confidence': 0.93,
      'date': '2025-11-04'
    },
    {
      'imagePath': 'assets/sample2.jpg',
      'label': 'nevus',
      'confidence': 0.88,
      'date': '2025-11-02'
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile & Scan History'),
        backgroundColor: Theme.of(context).colorScheme.primary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          children: [
            const SizedBox(height: 10),
            const Text(
              "Previous Scans",
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            Expanded(
              child: scanHistory.isEmpty
                  ? const Center(
                      child: Text(
                        "No scans yet. Try scanning a skin image first!",
                        style: TextStyle(color: Colors.grey, fontSize: 16),
                      ),
                    )
                  : ListView.builder(
                      itemCount: scanHistory.length,
                      itemBuilder: (context, index) {
                        final scan = scanHistory[index];
                        return Card(
                          elevation: 4,
                          margin: const EdgeInsets.symmetric(vertical: 8),
                          shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12)),
                          child: ListTile(
                            leading: ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: Image.file(
                                File(scan['imagePath']),
                                width: 60,
                                height: 60,
                                fit: BoxFit.cover,
                                errorBuilder: (context, error, stackTrace) {
                                  return Image.asset(
                                    'assets/placeholder.png',
                                    width: 60,
                                    height: 60,
                                  );
                                },
                              ),
                            ),
                            title: Text(
                              scan['label'].toString().toUpperCase(),
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            subtitle: Text(
                              "Confidence: ${(scan['confidence'] * 100).toStringAsFixed(1)}%\nDate: ${scan['date']}",
                            ),
                            trailing: IconButton(
                              icon: const Icon(Icons.delete, color: Colors.red),
                              onPressed: () {
                                setState(() {
                                  scanHistory.removeAt(index);
                                });
                              },
                            ),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
