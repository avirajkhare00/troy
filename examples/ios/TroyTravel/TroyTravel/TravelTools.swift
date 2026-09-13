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

    func add(day: Int, time: String, activity: String) {
        items.append(Item(day: day, time: time, activity: activity))
        items.sort { ($0.day, $0.time) < ($1.day, $1.time) }
    }
}

/// Mock travel tools. All data is fake and deterministic — this is a demo of
/// on-device tool calling, not a booking engine.
@MainActor
struct TravelTools {
    let itinerary: Itinerary

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

    func dispatch(_ call: ToolCall) -> String {
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
            itinerary.add(day: day, time: time, activity: activity)
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
