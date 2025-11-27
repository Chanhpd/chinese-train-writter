# Flutter Integration Guide - Handwriting Scoring API

## 📡 API Endpoints

### Base URL
```
http://your-server-ip:5000
```

---

## 🎯 Main Endpoint: `/score`

### Request
```dart
POST /score
Content-Type: application/json

{
  "character": "好",           // Optional - Chinese character for logging
  "image_reference": "base64", // Required - Base64 string of reference image
  "image_user": "base64"       // Required - Base64 string of user drawing
}
```

### Response
```dart
{
  "success": true,
  "character": "好",
  "distance": 0.2313,
  "score": 85.14,
  "interpretation": "Very Good",
  "timestamp": "2025-11-27T21:45:00"
}
```

### Score Interpretation
- **90-100**: Excellent ⭐⭐⭐⭐⭐
- **75-89**: Very Good ⭐⭐⭐⭐
- **60-74**: Good ⭐⭐⭐
- **40-59**: Fair ⭐⭐
- **0-39**: Poor ⭐

---

## 🔄 Batch Endpoint: `/batch_score`

### Request
```dart
POST /batch_score
Content-Type: application/json

{
  "items": [
    {
      "character": "好",
      "image_reference": "base64",
      "image_user": "base64"
    },
    {
      "character": "学",
      "image_reference": "base64",
      "image_user": "base64"
    }
  ]
}
```

### Response
```dart
{
  "success": true,
  "results": [
    {
      "character": "好",
      "distance": 0.2313,
      "score": 85.14,
      "interpretation": "Very Good"
    },
    {
      "character": "学",
      "distance": 0.3456,
      "score": 66.74,
      "interpretation": "Good"
    }
  ],
  "summary": {
    "total": 2,
    "average_score": 75.94
  },
  "timestamp": "2025-11-27T21:45:00"
}
```

---

## 💻 Flutter Code Example

### 1. Add Dependencies
```yaml
# pubspec.yaml
dependencies:
  http: ^1.1.0
  image: ^4.1.3
```

### 2. API Service Class
```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:image/image.dart' as img;

class HandwritingScoringService {
  final String baseUrl;
  
  HandwritingScoringService({required this.baseUrl});
  
  /// Score a single character
  Future<ScoringResult> scoreCharacter({
    required String character,
    required Uint8List referenceImage,
    required Uint8List userImage,
  }) async {
    try {
      // Convert images to base64
      final refBase64 = base64Encode(referenceImage);
      final userBase64 = base64Encode(userImage);
      
      // Make API request
      final response = await http.post(
        Uri.parse('$baseUrl/score'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'character': character,
          'image_reference': refBase64,
          'image_user': userBase64,
        }),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          return ScoringResult.fromJson(data);
        } else {
          throw Exception(data['error'] ?? 'Unknown error');
        }
      } else {
        throw Exception('HTTP ${response.statusCode}: ${response.body}');
      }
    } catch (e) {
      throw Exception('Failed to score character: $e');
    }
  }
  
  /// Score multiple characters at once
  Future<BatchScoringResult> scoreMultiple({
    required List<ScoringItem> items,
  }) async {
    try {
      final itemsJson = items.map((item) => {
        'character': item.character,
        'image_reference': base64Encode(item.referenceImage),
        'image_user': base64Encode(item.userImage),
      }).toList();
      
      final response = await http.post(
        Uri.parse('$baseUrl/batch_score'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'items': itemsJson}),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['success'] == true) {
          return BatchScoringResult.fromJson(data);
        } else {
          throw Exception(data['error'] ?? 'Unknown error');
        }
      } else {
        throw Exception('HTTP ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to score multiple: $e');
    }
  }
  
  /// Health check
  Future<bool> healthCheck() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['status'] == 'healthy' && data['model_loaded'] == true;
      }
      return false;
    } catch (e) {
      return false;
    }
  }
}

/// Data Models
class ScoringResult {
  final bool success;
  final String character;
  final double distance;
  final double score;
  final String interpretation;
  final String timestamp;
  
  ScoringResult({
    required this.success,
    required this.character,
    required this.distance,
    required this.score,
    required this.interpretation,
    required this.timestamp,
  });
  
  factory ScoringResult.fromJson(Map<String, dynamic> json) {
    return ScoringResult(
      success: json['success'],
      character: json['character'] ?? '',
      distance: json['distance'].toDouble(),
      score: json['score'].toDouble(),
      interpretation: json['interpretation'],
      timestamp: json['timestamp'],
    );
  }
}

class ScoringItem {
  final String character;
  final Uint8List referenceImage;
  final Uint8List userImage;
  
  ScoringItem({
    required this.character,
    required this.referenceImage,
    required this.userImage,
  });
}

class BatchScoringResult {
  final bool success;
  final List<ScoringResult> results;
  final int total;
  final double averageScore;
  
  BatchScoringResult({
    required this.success,
    required this.results,
    required this.total,
    required this.averageScore,
  });
  
  factory BatchScoringResult.fromJson(Map<String, dynamic> json) {
    return BatchScoringResult(
      success: json['success'],
      results: (json['results'] as List)
          .map((r) => ScoringResult.fromJson(r))
          .toList(),
      total: json['summary']['total'],
      averageScore: json['summary']['average_score'].toDouble(),
    );
  }
}
```

### 3. Usage Example
```dart
// Initialize service
final scoringService = HandwritingScoringService(
  baseUrl: 'http://your-server-ip:5000',
);

// Check if API is healthy
final isHealthy = await scoringService.healthCheck();
if (!isHealthy) {
  print('API is not ready');
  return;
}

// Score a character
final result = await scoringService.scoreCharacter(
  character: '好',
  referenceImage: referenceImageBytes,
  userImage: userDrawingBytes,
);

print('Score: ${result.score}/100');
print('Interpretation: ${result.interpretation}');
```

### 4. Get Image from Canvas
```dart
import 'dart:ui' as ui;
import 'package:flutter/rendering.dart';

Future<Uint8List> getCanvasImage(GlobalKey canvasKey) async {
  try {
    RenderRepaintBoundary boundary = 
        canvasKey.currentContext!.findRenderObject() as RenderRepaintBoundary;
    
    ui.Image image = await boundary.toImage(pixelRatio: 1.0);
    ByteData? byteData = await image.toByteData(
      format: ui.ImageByteFormat.png,
    );
    
    return byteData!.buffer.asUint8List();
  } catch (e) {
    throw Exception('Failed to capture canvas: $e');
  }
}
```

---

## 🚀 Running the API

### Local (Development)
```bash
python api_production.py
```
Access at: `http://localhost:5000`

### Production (with Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api_production:app
```

### Docker (Recommended for deployment)
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api_production:app"]
```

Build and run:
```bash
docker build -t handwriting-api .
docker run -p 5000:5000 handwriting-api
```

---

## 📝 Notes

1. **Image Format**: PNG or JPEG, base64 encoded
2. **Image Size**: Automatically resized to 128x128
3. **Timeout**: Default 30 seconds
4. **Rate Limiting**: Consider adding for production
5. **HTTPS**: Use reverse proxy (nginx) for production

---

## 🐛 Error Handling

### Common Errors
```dart
{
  "success": false,
  "error": "Model not loaded"
}

{
  "success": false,
  "error": "Missing required fields: image_reference, image_user"
}

{
  "success": false,
  "error": "Invalid base64 image data"
}
```

### Flutter Error Handling
```dart
try {
  final result = await scoringService.scoreCharacter(...);
  // Handle success
} on Exception catch (e) {
  if (e.toString().contains('Model not loaded')) {
    // Show "API is down" message
  } else if (e.toString().contains('HTTP 400')) {
    // Show "Invalid request" message
  } else {
    // Show generic error
  }
}
```
