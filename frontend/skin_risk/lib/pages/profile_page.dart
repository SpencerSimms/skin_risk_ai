import 'dart:io';
import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  List<Map<String, dynamic>> scanHistory = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    loadScanHistory();
  }

  Future<void> loadScanHistory() async {
    final user = Supabase.instance.client.auth.currentUser;
    if (user == null) {
      setState(() {
        _isLoading = false;
      });
      return;
    }

    try {
      final response = await Supabase.instance.client
          .from('scans')
          .select()
          .eq('user_id', user.id)
          .order('created_at', ascending: false);

      setState(() {
        scanHistory = List<Map<String, dynamic>>.from(response);
        _isLoading = false;
      });
    } catch (e) {
      debugPrint("Error loading scans: $e");
      setState(() => _isLoading = false);
    }
  }

  Future<void> deleteScan(String id, String fileName) async {
    try {
      final supabase = Supabase.instance.client;
      final user = supabase.auth.currentUser;
  
      if (user == null) {
        debugPrint("No logged-in user found.");
        return;
      }
  
      // 1. Delete from storage bucket
      final storageResponse = await supabase.storage.from('scans').remove([fileName]);
  
      if (storageResponse.isEmpty) {
        debugPrint("Warning: File $fileName was not found in bucket.");
      } else {
        debugPrint("File $fileName deleted from bucket.");
      }
  
      // 2. Delete row from table
      await supabase.from('scans').delete().eq('id', id);
      debugPrint("Row deleted successfully.");
  
      // 3. Update local state
      setState(() {
        scanHistory.removeWhere((scan) => scan['id'] == id);
      });
    } catch (e) {
      debugPrint("Error deleting scan: $e");
    }
  }

  Future<void> _logout() async {
    await Supabase.instance.client.auth.signOut();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('You have been logged out.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile & Scan History'),
        backgroundColor: Theme.of(context).colorScheme.primary,
        actions: [
          IconButton(
            icon: const Icon(Icons.logout, color: Colors.white),
            tooltip: 'Logout',
            onPressed: _logout,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : scanHistory.isEmpty
              ? const Center(
                  child: Text(
                    "No scans yet. Try scanning a skin image first!",
                    style: TextStyle(color: Colors.grey, fontSize: 16),
                  ),
                )
              : Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: ListView.builder(
                    itemCount: scanHistory.length,
                    itemBuilder: (context, index) {
                      final scan = scanHistory[index];
                      final fileName = scan['file_name'];
                      final id = scan['id'];

                      return Card(
                        elevation: 4,
                        margin: const EdgeInsets.symmetric(vertical: 8),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: ListTile(
                          leading: ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: scan['image_url'] != null
                                ? Image.network(
                                    scan['image_url'],
                                    width: 60,
                                    height: 60,
                                    fit: BoxFit.cover,
                                    errorBuilder:
                                        (context, error, stackTrace) =>
                                            Image.asset(
                                      'assets/placeholder.png',
                                      width: 60,
                                      height: 60,
                                    ),
                                  )
                                : Image.asset(
                                    'assets/placeholder.png',
                                    width: 60,
                                    height: 60,
                                  ),
                          ),
                          title: Text(
                            scan['prediction']?.toString().toUpperCase() ??
                                'UNKNOWN',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          subtitle: Text(
                            "Confidence: ${(scan['confidence'] * 100).toStringAsFixed(1)}%\nDate: ${scan['created_at'] ?? 'N/A'}",
                          ),
                          trailing: IconButton(
                            icon: const Icon(Icons.delete, color: Colors.red),
                            onPressed: () async {
                              if (fileName != null && fileName.isNotEmpty) {
                                await deleteScan(id, fileName);
                              } else {
                                debugPrint(
                                    "No file name found for this scan.");
                              }
                            },
                          ),
                        ),
                      );
                    },
                  ),
                ),
    );
  }
}
