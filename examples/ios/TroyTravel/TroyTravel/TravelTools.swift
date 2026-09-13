import Foundation
import MLXLMCommon

/// The trip being planned — the model builds this through tool calls.
@Observable
@MainActor
final class Itinerary {
    struct Item: Identifiable {
        let id = UUID()
        let day: Int
        let time: String
        let activity: String
    }

    private(set) var items: [Item] = []

    /// Returns false for an exact duplicate instead of adding it twice.
    @discardableResult
    func add(day: Int, time: String, activity: String) -> Bool {
        let duplicate = items.contains {
            $0.day == day && $0.activity.caseInsensitiveCompare(activity) == .orderedSame
        }
        guard !duplicate else { return false }
        items.append(Item(day: day, time: time, activity: activity))
        items.sort { ($0.day, $0.time) < ($1.day, $1.time) }
        return true
    }
}

/// Mock travel tools. All data is fake and deterministic — this is a demo of
/// on-device tool calling, not a booking engine.
/// Thrown when the model keeps calling tools after its budget ran out.
struct ToolLimitReached: Error {}

@MainActor
final class TravelTools {
    let itinerary: Itinerary
    private var callsThisTurn = 0
    private var seenCalls = Set<String>()
    private let maxCallsPerTurn = 6

    init(itinerary: Itinerary) {
        self.itinerary = itinerary
    }

    /// Call at the start of each user turn.
    func resetTurn() {
        callsThisTurn = 0
        seenCalls.removeAll()
    }

    var specs: [ToolSpec] {
        [
            spec(
                name: "search_flights",
                description: "Search for flights between two cities on a date.",
                properties: [
                    "from": ["type": "string", "description": "Departure city"],
                    "to": ["type": "string", "description": "Destination city"],
                    "date": ["type": "string", "description": "Date, YYYY-MM-DD"],
                ],
                required: ["from", "to"]),
            spec(
                name: "find_hotels",
                description: "Find hotels in a city within a nightly budget.",
                properties: [
                    "city": ["type": "string", "description": "City to search"],
                    "max_price_usd": ["type": "number", "description": "Max nightly price in USD"],
                ],
                required: ["city"]),
            spec(
                name: "get_weather",
                description: "Get the weather forecast for a city.",
                properties: [
                    "city": ["type": "string", "description": "City name"]
                ],
                required: ["city"]),
            spec(
                name: "add_to_itinerary",
                description: "Add one activity to the trip itinerary.",
                properties: [
                    "day": ["type": "integer", "description": "Trip day number, starting at 1"],
                    "time": ["type": "string", "description": "Time of day, e.g. 09:00"],
                    "activity": ["type": "string", "description": "Short activity description"],
                ],
                required: ["day", "activity"]),
        ]
    }

    func dispatch(_ call: ToolCall) throws -> String {
        // Guardrails for a small model: budget per turn, and reject repeats —
        // both break tool-call loops by telling the model to summarize.
        callsThisTurn += 1
        if callsThisTurn > maxCallsPerTurn {
            // One warning the model can act on; if it keeps calling anyway,
            // hard-stop the turn rather than looping on error messages.
            if callsThisTurn > maxCallsPerTurn + 1 {
                throw ToolLimitReached()
            }
            return #"{"error": "tool budget exhausted — stop calling tools and summarize the plan for the user now"}"#
        }
        // Deterministic signature: sorted-keys JSON of the call, so identical
        // calls always collide (dictionary interpolation order is not stable).
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let signature = (try? encoder.encode(call.function)).map {
            String(decoding: $0, as: UTF8.self)
        } ?? "\(call.function.name)"
        if !seenCalls.insert(signature).inserted {
            return #"{"error": "duplicate call — this was already done, do not repeat it; summarize for the user"}"#
        }

        let args = call.function.arguments
        func str(_ key: String) -> String {
            if case .string(let s)? = args[key] { return s }
            return ""
        }
        func num(_ key: String, default fallback: Double) -> Double {
            switch args[key] {
            case .double(let d)?: return d
            case .int(let i)?: return Double(i)
            default: return fallback
            }
        }

        switch call.function.name {
        case "search_flights":
            let from = str("from"), to = str("to")
            return """
                {"flights": [
                  {"airline": "Vistara", "depart": "\(from) 08:10", "arrive": "\(to) 10:45", "price_usd": 142},
                  {"airline": "IndiGo", "depart": "\(from) 13:30", "arrive": "\(to) 16:05", "price_usd": 98},
                  {"airline": "Air India", "depart": "\(from) 19:20", "arrive": "\(to) 21:55", "price_usd": 121}
                ]}
                """
        case "find_hotels":
            let city = str("city")
            let cap = Int(num("max_price_usd", default: 200))
            return """
                {"hotels": [
                  {"name": "The \(city) Grand", "rating": 4.6, "price_usd": \(min(cap, 180))},
                  {"name": "Old Town Inn", "rating": 4.2, "price_usd": \(min(cap, 95))},
                  {"name": "Riverside Stay", "rating": 4.4, "price_usd": \(min(cap, 120))}
                ]}
                """
        case "get_weather":
            return """
                {"city": "\(str("city"))", "forecast": [
                  {"day": 1, "condition": "sunny", "high_c": 28, "low_c": 19},
                  {"day": 2, "condition": "partly cloudy", "high_c": 26, "low_c": 18},
                  {"day": 3, "condition": "light rain", "high_c": 23, "low_c": 17}
                ]}
                """
        case "add_to_itinerary":
            let day = Int(num("day", default: 1))
            let time = str("time").isEmpty ? "09:00" : str("time")
            let activity = str("activity")
            guard itinerary.add(day: day, time: time, activity: activity) else {
                return #"{"error": "already on the itinerary for day \#(day) — do not add it again; summarize for the user"}"#
            }
            return #"{"status": "added", "day": \#(day), "activity": "\#(activity)"}"#
        default:
            return #"{"error": "unknown tool \#(call.function.name)"}"#
        }
    }

    private func spec(
        name: String, description: String,
        properties: [String: [String: any Sendable]], required: [String]
    ) -> ToolSpec {
        [
            "type": "function",
            "function": [
                "name": name,
                "description": description,
                "parameters": [
                    "type": "object",
                    "properties": properties,
                    "required": required,
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    }
}
