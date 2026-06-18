// ─────────────────────────────────────────────────────────────────────────────
//  A2UI 卡片渲染器：负责 form_card 与 place_card
//  依赖 app.js 中定义的 WebSocket、地图和会话持久化函数，因此需后加载。
// ─────────────────────────────────────────────────────────────────────────────

window.A2UICards = (function () {
  "use strict";

  // ── DOM 辅助函数 ───────────────────────────────────────────────────────

  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function el(tag, className, attrs) {
    // 文本统一使用 textContent 写入，避免地点/API 数据被当作 HTML 执行。
    var e = document.createElement(tag);
    if (className) { e.className = className; }
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === "text") { e.textContent = attrs[k]; return; }
        e.setAttribute(k, attrs[k]);
      });
    }
    return e;
  }

  // ── 信息补充表单卡片 ──────────────────────────────────────────────────

  function renderFormCard(payload) {
    var card = el("div", "a2ui-card a2ui-form-card", { "data-a2ui-id": payload.id || "" });

    // 卡片头部
    var header = el("div", "a2ui-card-header");
    header.appendChild(el("span", "a2ui-card-icon", { text: "\u{1F4DD}" }));
    header.appendChild(el("span", "a2ui-card-title", { text: payload.title || "完善旅行信息" }));
    card.appendChild(header);

    // 根据后端下发的字段 Schema 动态构造表单主体。
    var body = el("div", "a2ui-card-body");
    if (payload.message) {
      body.appendChild(el("p", "a2ui-card-msg", { text: payload.message }));
    }

    var fieldsWrap = el("div", "a2ui-form-fields");
    (payload.fields || []).forEach(function (f) {
      var field = el("div", "a2ui-field");
      var label = el("label", "a2ui-field-label");
      label.textContent = (f.label || f.key);
      if (f.required) {
        var star = el("span", "a2ui-required", { text: " *" });
        label.appendChild(star);
      }
      field.appendChild(label);

      var input;
      if (f.field_type === "select") {
        input = el("select", "a2ui-field-select", { name: f.key });
        var emptyOpt = document.createElement("option");
        emptyOpt.value = "";
        emptyOpt.textContent = "请选择…";
        input.appendChild(emptyOpt);
        (f.options || []).forEach(function (o) {
          var opt = document.createElement("option");
          opt.value = o.value;
          opt.textContent = o.label;
          input.appendChild(opt);
        });
      } else if (f.field_type === "textarea") {
        input = el("textarea", "a2ui-field-textarea", {
          name: f.key,
          placeholder: f.placeholder || "",
          rows: "2",
        });
      } else {
        input = el("input", "a2ui-field-input", {
          name: f.key,
          type: f.field_type || "text",
          placeholder: f.placeholder || "",
        });
        if (f.min != null) { input.setAttribute("min", f.min); }
        if (f.max != null) { input.setAttribute("max", f.max); }
      }
      if (f.required) { input.setAttribute("required", ""); }
      field.appendChild(input);
      fieldsWrap.appendChild(field);
    });
    body.appendChild(fieldsWrap);
    card.appendChild(body);

    // 操作区
    var footer = el("div", "a2ui-card-footer");
    var submitBtn = el("button", "a2ui-btn a2ui-btn-submit", { text: "发送" });
    var cancelBtn = el("button", "a2ui-btn a2ui-btn-cancel", { text: "取消" });

    var cardId = payload.id || "";
    submitBtn.addEventListener("click", function () {
      _submitForm(cardId, card);
    });
    cancelBtn.addEventListener("click", function () {
      card.remove();
    });

    footer.appendChild(submitBtn);
    footer.appendChild(cancelBtn);
    card.appendChild(footer);

    return card;
  }

  function _submitForm(cardId, formEl) {
    var data = {};
    var inputs = formEl.querySelectorAll("[name]");
    var emptyRequired = null;

    inputs.forEach(function (inp) {
      var val = inp.value.trim();
      if (val) { data[inp.name] = inp.value; }
      else if (inp.hasAttribute("required") && !emptyRequired) { emptyRequired = inp; }
    });

    if (emptyRequired) {
      var labelEl = emptyRequired.closest(".a2ui-field");
      var labelText = labelEl ? labelEl.querySelector(".a2ui-field-label").textContent : "该项";
      alert("请填写：" + labelText.replace(/\s*\*$/, ""));
      return;
    }

    // 提交后锁定表单，防止用户重复发送同一响应。
    var submitBtn = formEl.querySelector(".a2ui-btn-submit");
    if (submitBtn) { submitBtn.disabled = true; submitBtn.textContent = "已发送"; }
    var allInputs = formEl.querySelectorAll("input, select, textarea");
    allInputs.forEach(function (inp) { inp.disabled = true; });

    // 通过 app.js 的全局 WebSocket 发送结构化响应。
    var a2uiMsg = "@@A2UI@@" + JSON.stringify({ type: "form_response", id: cardId, data: data });
    if (typeof ws !== "undefined" && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(a2uiMsg);
    }

    // 同步保存到浏览器会话，便于刷新后重放 UI。
    if (typeof _saveA2UICard === "function") {
      _saveA2UICard({ type: "form_response", id: cardId, data: data });
    }
  }

  // ── 地点卡片 ──────────────────────────────────────────────────────────

  function renderPlaceCard(payload) {
    var places = payload.places || [];
    if (places.length === 0) { return el("div", "a2ui-card", { text: "" }); }

    // If multiple places, wrap in a container
    var wrapper = document.createElement("div");

    places.forEach(function (p) {
      var card = el("div", "a2ui-card a2ui-place-card", { "data-place-id": p.id || "" });

      // Photo
      var photoWrap = el("div", "a2ui-place-photo-wrap");
      if (p.photos && p.photos[0]) {
        var img = document.createElement("img");
        img.className = "a2ui-place-photo";
        img.src = p.photos[0];
        img.loading = "lazy";
        img.onerror = function () { this.style.display = "none"; };
        photoWrap.appendChild(img);
      }
      if (p.label) {
        photoWrap.appendChild(el("span", "a2ui-place-category", { text: p.label }));
      }
      card.appendChild(photoWrap);

      // Body
      var body = el("div", "a2ui-place-body");
      body.appendChild(el("div", "a2ui-place-name", { text: p.name }));
      if (p.summary || p.rating || p.cost || p.cuisine) {
        var meta = el("div", "a2ui-place-meta");
        if (p.rating) { meta.appendChild(el("span", "", { text: "⭐ " + p.rating })); }
        if (p.cost)   { meta.appendChild(el("span", "", { text: "💰 ¥" + p.cost })); }
        if (p.cuisine) { meta.appendChild(el("span", "", { text: "🍽️ " + p.cuisine })); }
        body.appendChild(meta);
      }
      if (p.address) {
        body.appendChild(el("div", "a2ui-place-addr", { text: "📍 " + p.address }));
      }
      if (p.summary) {
        body.appendChild(el("div", "a2ui-place-summary", { text: p.summary }));
      }
      card.appendChild(body);

      // Actions
      var actions = el("div", "a2ui-place-actions");
      if (p.longitude != null && p.latitude != null) {
        var focusBtn = el("button", "a2ui-btn a2ui-btn-focus", { text: "🗺️ 查看地图" });
        focusBtn.addEventListener("click", function () {
          if (typeof focusPoi === "function") {
            focusPoi(p.longitude, p.latitude, p.name, p.category || "poi", p);
          }
        });
        actions.appendChild(focusBtn);
      }
      var addBtn = el("button", "a2ui-btn a2ui-btn-add", { text: "➕ 加入行程" });
      addBtn.addEventListener("click", function () {
        _addPlaceToPlan(p);
        addBtn.textContent = "✓ 已加入";
        addBtn.disabled = true;
        addBtn.classList.add("a2ui-btn-added");
      });
      actions.appendChild(addBtn);
      card.appendChild(actions);

      wrapper.appendChild(card);
    });

    return wrapper.children.length === 1 ? wrapper.firstChild : wrapper;
  }

  function _addPlaceToPlan(placeData) {
    window._a2uiSavedPlaces = window._a2uiSavedPlaces || [];
    if (window._a2uiSavedPlaces.some(function (s) { return s.id === placeData.id; })) {
      return;
    }
    window._a2uiSavedPlaces.push(placeData);
  }

  // ── Public API ─────────────────────────────────────────────────────────

  return {
    renderFormCard: renderFormCard,
    renderPlaceCard: renderPlaceCard,
  };
})();
