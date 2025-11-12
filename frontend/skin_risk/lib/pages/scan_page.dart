import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';

class ScanPage extends StatefulWidget {
  const ScanPage({super.key});

  @override
  State<ScanPage> createState() => _ScanPageState();
}

class _ScanPageState extends State<ScanPage> {
  final picker = ImagePicker();
  File? _image;
  String? _prediction;
  String? _confidence;

  final supabase = Supabase.instance.client;
  final String userId = Supabase.instance.client.auth.currentUser?.id ?? "";

  Future<void> _getImage(ImageSource source) async {
    final pickedFile = await picker.pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _prediction = null;
        _confidence = null;
      });
      await _uploadImage(_image!);
    }
  }

  Future<void> _uploadImage(File imageFile) async {
    final uri = Uri.parse('http://10.0.2.2:8000/predict'); // local backend
    final request = http.MultipartRequest('POST', uri);

    request.fields['user_id'] = userId;
    request.files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    final response = await request.send();

    if (response.statusCode == 200) {
      final respStr = await response.stream.bytesToString();
      final data = jsonDecode(respStr);

      setState(() {
        _prediction = data['predicted_label'];
        _confidence = double.parse(data['confidence'].toString()).toStringAsFixed(3);
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Error uploading image')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Skin Risk Scan')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            if (_image != null)
              Image.file(_image!, height: 200, fit: BoxFit.cover)
            else
              const Icon(Icons.image, size: 200, color: Colors.grey),

            const SizedBox(height: 16),

            ElevatedButton.icon(
              onPressed: () => _getImage(ImageSource.camera),
              icon: const Icon(Icons.camera_alt),
              label: const Text('Take Photo'),
            ),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              onPressed: () => _getImage(ImageSource.gallery),
              icon: const Icon(Icons.photo_library),
              label: const Text('Select from Files'),
            ),

            const SizedBox(height: 20),

            if (_prediction != null)
              Card(
                color: Colors.blue.shade50,
                child: ListTile(
                  title: Text('Prediction: $_prediction'),
                  subtitle: Text('Confidence: $_confidence'),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
