# Ground Truth Justifications

## command_injection_ping_001
This case is labeled `CommandInjection` because the submitted IP value contains chained shell commands and the response returns direct command output such as `uname`, directory listings, the working directory, the current user, and network data. The severity is `High` because the application is clearly executing attacker-controlled system commands on the server, which can lead to broad system enumeration and further compromise.

## cookie_security_attribute_observation_001
This case is labeled `Other` because the packet shows a cookie with `HttpOnly=false` and `Secure=false`, which is a real security weakness, but it does not by itself prove a more specific issue such as session prediction or direct takeover. The severity is `Low` because this is a hardening failure with plausible risk, but the evidence does not show active exploitation or immediate high-impact abuse.

## hard_cookie_md5_002
This case is labeled `WeakSessionIDs` because the observed cookie values match deterministic MD5 hashes of a simple sequence, which makes the session tokens predictable rather than random. The severity is `High` because predictable session identifiers can let an attacker guess valid sessions and take over user accounts.

## medium_cookie_timestamps_001
This case is labeled `WeakSessionIDs` because the session cookie values follow a clear timestamp-like progression and are therefore guessable instead of unpredictable. The severity is `High` because predictable session identifiers can be enumerated and used for session hijacking.

## reflected_input_password_change_guard_001
This case is labeled `CSRF` because the password change flow relies on a referer check, and the packet shows attacker-controlled script execution on an application page that can generate same-origin requests which satisfy that guard. The severity is `High` because the affected action is password change for an authenticated user, which directly impacts account control.

## info_apache
This case is labeled `NoFinding` because the packet only shows normal response metadata such as a redirect, common cache headers, and a generic `Server: Apache` header, without evidence of an actual vulnerability. The severity is `Info` because the observation is informational only and does not demonstrate exploitable behavior.

## error_message_path_disclosure_001
This case is labeled `FileInclusion` because the request uses a `page=` parameter and the response shows a failing `include()` call with an explicit include path, which points to an include sink rather than a simple disclosure-only condition. The severity is `High` because file inclusion can lead to sensitive file access and potentially severe application compromise.
