import Foundation

/// Thin JSON client for the `troy mesh serve` coordinator API.
struct MeshClient {
    let baseURL: URL
    let token: String

    struct WorkItem: Decodable {
        let id: String
        let prompt: String
        let maxTokens: Int
        let temperature: Double
        let leaseSeconds: Double
    }

    struct WorkResponse: Decodable {
        let done: Bool
        let retrySeconds: Double?
        let items: [WorkItem]
    }

    struct ResultsResponse: Decodable {
        let accepted: Int
        let duplicates: Int
        let records: Int
        let target: Int
        let done: Bool
    }

    enum MeshError: LocalizedError {
        case badStatus(Int)

        var errorDescription: String? {
            switch self {
            case .badStatus(401): "Coordinator rejected the token."
            case .badStatus(let code): "Coordinator returned HTTP \(code)."
            }
        }
    }

    private func request(_ path: String, body: Data? = nil, timeout: TimeInterval) -> URLRequest {
        var req = URLRequest(url: baseURL.appendingPathComponent(path), timeoutInterval: timeout)
        req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        if let body {
            req.httpMethod = "POST"
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return req
    }

    private func send<T: Decodable>(_ req: URLRequest) async throws -> T {
        let (data, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw MeshError.badStatus(http.statusCode)
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(T.self, from: data)
    }

    func fetchWork(worker: String, max: Int = 1) async throws -> WorkResponse {
        var req = request("v1/work", timeout: 10)
        req.url = req.url.flatMap {
            var parts = URLComponents(url: $0, resolvingAgainstBaseURL: false)
            parts?.queryItems = [URLQueryItem(name: "max", value: String(max))]
            return parts?.url
        }
        req.setValue(worker, forHTTPHeaderField: "X-Worker")
        return try await send(req)
    }

    func postResult(worker: String, id: String, raw: String) async throws -> ResultsResponse {
        let body = try JSONSerialization.data(withJSONObject: [
            "worker": worker,
            "results": [["id": id, "raw": raw]],
        ])
        return try await send(request("v1/results", body: body, timeout: 30))
    }
}
